"""
logiprune.entropy_pipeline
──────────────────────────
sklearn-compatible LogiPruneEntropy estimator (Paper 2).

Wraps EntropyAnalyzer in a fit/transform/pruned_grid API
that mirrors LogiPrune (Paper 1), enabling drop-in use and
seamless combination in the two-stage pipeline.
"""

import numpy as np
import pandas as pd
import time
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from .discretize import AdaptiveDiscretizer
from .sweeper    import SWTSSweeper
from .relations  import BICONDITIONAL
from .entropy    import EntropyAnalyzer


class LogiPruneEntropy:
    """
    A priori model complexity selection via truth table entropy (Paper 2).

    Computes the minimum Shannon entropy H* of the 4-cell truth table
    distribution across all feature pairs and threshold sweeps, and uses
    H* to restrict the XGBoost (or any tree-based) hyperparameter grid
    before search begins.

    Also performs validated feature elimination (biconditional pairs with
    accuracy drop ≤ acc_drop_tolerance) with a feedback loop: eliminated
    features are reinstated if their removal increases H(feature, target)
    by more than feedback_delta.

    Parameters
    ----------
    base_grid : dict
        Full hyperparameter grid. Keys should match the estimator step name
        (e.g. 'xgb__max_depth' for a Pipeline with step named 'xgb').
    acc_drop_tolerance : float, default=0.04
        Max accuracy drop allowed when eliminating a biconditional feature.
    feedback_delta : float, default=0.10
        Entropy increase threshold that triggers feature reinstatement.
    discretizer_strategy : str, default='percentile'
        Normalization strategy for AdaptiveDiscretizer.
    cv : int, default=3
        Cross-validation folds for elimination validation.
    verbose : bool, default=False

    Attributes
    ----------
    entropy_analyzer_ : EntropyAnalyzer
        Fitted entropy analyzer with all profiles.
    eliminated_features_ : list of str
        Features removed via validated biconditional rule.
    reinstated_features_ : list of dict
        Features that were reinstated by the feedback loop.
    pruned_grid_ : dict
        Pruned hyperparameter grid ready for GridSearchCV.
    preprocessing_time_ : float
    """

    def __init__(
        self,
        base_grid: dict,
        acc_drop_tolerance: float = 0.04,
        feedback_delta: float = 0.10,
        discretizer_strategy: str = 'percentile',
        cv: int = 3,
        verbose: bool = False,
    ):
        self.base_grid            = base_grid
        self.acc_drop_tolerance   = acc_drop_tolerance
        self.feedback_delta       = feedback_delta
        self.discretizer_strategy = discretizer_strategy
        self.cv                   = cv
        self.verbose              = verbose

        self.entropy_analyzer_:    Optional[EntropyAnalyzer]    = None
        self.eliminated_features_: list = []
        self.reinstated_features_: list = []
        self.pruned_grid_:         dict = {}
        self.preprocessing_time_:  float = 0.0
        self._discretizer:         Optional[AdaptiveDiscretizer] = None
        self._fitted:              bool = False

    def _log(self, msg):
        if self.verbose:
            print(f"[LogiPruneEntropy] {msg}")

    def _validate_elimination(self, Xn: pd.DataFrame,
                               col_remove: str, y: pd.Series) -> bool:
        cols_w  = Xn.columns.tolist()
        cols_wo = [c for c in cols_w if c != col_remove]
        if not cols_wo:
            return False
        cv_obj = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        pipe   = Pipeline([('sc', StandardScaler()),
                           ('lr', LogisticRegression(max_iter=300, random_state=0))])
        a_w  = cross_val_score(pipe, Xn[cols_w],  y, cv=cv_obj).mean()
        a_wo = cross_val_score(pipe, Xn[cols_wo], y, cv=cv_obj).mean()
        return (a_w - a_wo) <= self.acc_drop_tolerance

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LogiPruneEntropy':
        t0 = time.time()

        # Step 1: normalize
        self._discretizer = AdaptiveDiscretizer(strategy=self.discretizer_strategy)
        Xn = self._discretizer.fit_transform(X)

        cv_obj = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        # Step 2: biconditional feature elimination with validation
        sw = SWTSSweeper(min_confidence=0.75)
        sw.fit(Xn)
        removed = set()
        for (ca, cb), r in sw.results_.items():
            if r.relation == BICONDITIONAL and r.confidence >= 0.90 and len(removed) < 6:
                if ca not in removed and cb not in removed:
                    col_rm = cb if Xn[ca].var() >= Xn[cb].var() else ca
                    if self._validate_elimination(Xn, col_rm, y):
                        removed.add(col_rm)
                        self._log(f"Eliminated: {col_rm} (biconditional with conf={r.confidence:.3f})")

        # Step 3: entropy analysis
        self.entropy_analyzer_ = EntropyAnalyzer(
            feedback_delta=self.feedback_delta)
        self.entropy_analyzer_.fit(Xn, y)
        self._log(self.entropy_analyzer_.summary())

        # Step 4: feedback loop — reinstate features if entropy increases
        if removed:
            feedback = self.entropy_analyzer_.feedback_check(list(removed), Xn, y)
            for item in feedback:
                self._log(f"Reinstating {item['col']}: {item['reason']}")
                removed.discard(item['col'])
                self.reinstated_features_.append(item)

        self.eliminated_features_ = list(removed)

        # Step 5: grid restriction from entropy
        self.pruned_grid_ = self.entropy_analyzer_.xgb_grid_from_entropy(
            self.base_grid)

        self.preprocessing_time_ = round(time.time() - t0, 3)
        self._fitted = True
        self._Xn = Xn   # store for transform
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        keep = [c for c in X.columns if c not in self.eliminated_features_]
        return X[keep]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def pruned_grid(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before pruned_grid().")
        return self.pruned_grid_

    def savings_summary(self) -> dict:
        if not self._fitted:
            raise RuntimeError("Call fit() before savings_summary().")
        def count(g):
            t = 1
            for v in g.values(): t *= len(v)
            return t
        cb  = count(self.base_grid)
        clp = count(self.pruned_grid_)
        ea  = self.entropy_analyzer_
        return {
            'configs_base':      cb,
            'configs_pruned':    clp,
            'config_savings_pct': round((cb - clp) / cb * 100, 1),
            'features_eliminated': len(self.eliminated_features_),
            'features_reinstated': len(self.reinstated_features_),
            'h_min':             ea.dataset_h_min_,
            'h_mean':            ea.dataset_h_mean_,
            'h_applied':         round(getattr(ea, '_applied_h', ea.dataset_h_min_), 3),
            'preprocessing_time_s': self.preprocessing_time_,
        }

    def report(self) -> str:
        if not self._fitted:
            raise RuntimeError("Call fit() before report().")
        s = self.savings_summary()
        lines = [
            "\n══════════════════════════════════════════════════",
            "  LogiPruneEntropy Report (Paper 2)",
            "══════════════════════════════════════════════════",
            f"  Preprocessing time:    {s['preprocessing_time_s']:.2f}s",
            f"  H_min (pairs):         {s['h_min']}",
            f"  H_mean (pairs):        {s['h_mean']}",
            f"  H_applied (weighted):  {s['h_applied']}",
            f"  Complexity class:      "
            f"{'very_low' if s['h_applied']<0.5 else 'low' if s['h_applied']<1.0 else 'medium' if s['h_applied']<1.5 else 'high'}",
            "",
            f"  Configs (base):        {s['configs_base']}",
            f"  Configs (pruned):      {s['configs_pruned']}",
            f"  Config savings:        {s['config_savings_pct']}%",
            "",
            f"  Features eliminated:   {s['features_eliminated']}  {self.eliminated_features_}",
            f"  Features reinstated:   {s['features_reinstated']}",
        ]
        if self.reinstated_features_:
            for item in self.reinstated_features_:
                lines.append(f"    ↺ {item['col']}: ΔH={item['delta']:.3f}")
        lines.append("══════════════════════════════════════════════════")
        return "\n".join(lines)
