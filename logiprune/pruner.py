"""
logiprune.pruner
────────────────
Translates the output of IterativeCoverage into concrete
GridSearch parameter grids — one per layer plus one for
the uncovered residual.

The pruner also validates feature elimination using a
fast LogisticRegression proxy before accepting any
biconditional-based removal. This is the FIX 1 from v2,
now integrated properly with the layer structure.

Key principle (the propositional chain):
  A↔B     (feature equivalence)
  A→D     (feature implies target, from Layer 2 analysis)
  ─────────────────────────────────────────────
  ∴ B is redundant w.r.t. D    (transitive pruning)

Without both premises, B is NOT eliminated.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from .coverage import IterativeCoverage, LogicLayer
from .relations import BICONDITIONAL, A_IMPLIES_B, B_IMPLIES_A, INCOMPATIBLE


class GridPruner:
    """
    Given a fitted IterativeCoverage and a base param_grid,
    produces a pruned grid for each layer.

    Parameters
    ----------
    base_grid : dict
        The full GridSearch parameter grid (e.g. for SVC pipeline).
    acc_drop_tolerance : float
        Max accuracy drop allowed when eliminating a feature (default 0.04).
    min_mi_for_elimination : float
        A feature is only eliminated if the surviving feature has
        MI with target ≥ this value (default 0.05).
        Prevents eliminating features when neither has target signal.
    cv : int
        Folds for validation cross-val (default 3).
    """

    def __init__(self,
                 base_grid: dict,
                 acc_drop_tolerance: float = 0.04,
                 min_mi_for_elimination: float = 0.05,
                 cv: int = 3):
        self.base_grid   = dict(base_grid)
        self.acc_tol     = acc_drop_tolerance
        self.min_mi      = min_mi_for_elimination
        self.cv          = cv

        # Results after fit
        self.layer_grids_:    list[dict] = []       # one per layer
        self.residual_grid_:  dict = {}
        self.features_to_remove_: set = set()
        self.validation_log_: list = []

    # ── Validation gate ──────────────────────────────────────────────────────

    def _validate_elimination(self,
                               X_norm: pd.DataFrame,
                               col_to_remove: str,
                               col_to_keep: str,
                               y: pd.Series,
                               mi_scores: dict) -> bool:
        """
        Accept elimination only if:
        1. The surviving feature has MI with target ≥ min_mi (target signal exists)
        2. Removing col_to_remove does not drop accuracy by more than acc_tol
        """
        # Gate 1: target signal in surviving feature
        if mi_scores.get(col_to_keep, 0) < self.min_mi:
            self.validation_log_.append({
                'col': col_to_remove, 'decision': 'KEEP',
                'reason': f'surviving MI={mi_scores.get(col_to_keep,0):.4f} < {self.min_mi}'
            })
            return False

        # Gate 2: accuracy drop check via LR proxy
        cv_obj = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        pipe   = Pipeline([('sc', StandardScaler()),
                           ('lr', LogisticRegression(max_iter=300, random_state=0))])
        cols_all = X_norm.columns.tolist()

        acc_with    = cross_val_score(pipe, X_norm[cols_all],
                                      y, cv=cv_obj, scoring='accuracy').mean()
        cols_pruned = [c for c in cols_all if c != col_to_remove]
        acc_without = cross_val_score(pipe, X_norm[cols_pruned],
                                      y, cv=cv_obj, scoring='accuracy').mean()
        drop = acc_with - acc_without

        decision = 'REMOVE' if drop <= self.acc_tol else 'KEEP'
        self.validation_log_.append({
            'col': col_to_remove, 'decision': decision,
            'acc_with': round(acc_with, 4),
            'acc_without': round(acc_without, 4),
            'drop': round(drop, 4),
            'tolerance': self.acc_tol
        })
        return decision == 'REMOVE'

    # ── Grid construction ────────────────────────────────────────────────────

    def _merge_restrictions(self, base: dict, restrictions: dict) -> dict:
        """Apply layer restrictions on top of base grid."""
        g = dict(base)
        for k, v in restrictions.items():
            if k.startswith('svc__') and k in g:
                # Intersect: only keep values present in both
                merged = [x for x in v if x in g[k]]
                if merged:   # only apply if intersection non-empty
                    g[k] = merged
        return g

    # ── Main fit ─────────────────────────────────────────────────────────────

    def fit(self,
            coverage: IterativeCoverage,
            X_norm: pd.DataFrame,
            y: pd.Series) -> 'GridPruner':
        """
        Parameters
        ----------
        coverage : fitted IterativeCoverage
        X_norm   : normalized training features (output of discretizer)
        y        : training target
        """
        self.features_to_remove_ = set()
        self.layer_grids_ = []
        self.validation_log_ = []

        # Compute MI scores once (fast, O(n×C))
        mi_vals = mutual_info_classif(X_norm, y, random_state=0)
        mi_scores = dict(zip(X_norm.columns, mi_vals))

        for layer in coverage.layers_:
            restr = layer.grid_restrictions
            grid  = self._merge_restrictions(self.base_grid, restr)

            # Handle feature elimination for biconditionals
            if restr.get('action') == 'eliminate_one':
                ca, cb = layer.col_a, layer.col_b

                # Choose which to keep: higher MI with target
                if mi_scores.get(ca, 0) >= mi_scores.get(cb, 0):
                    col_keep, col_remove = ca, cb
                else:
                    col_keep, col_remove = cb, ca

                # Validate elimination
                if col_remove not in self.features_to_remove_:
                    safe = self._validate_elimination(
                        X_norm, col_remove, col_keep, y, mi_scores)
                    if safe:
                        self.features_to_remove_.add(col_remove)

            self.layer_grids_.append({
                'layer_id':   layer.layer_id,
                'relation':   layer.relation,
                'col_a':      layer.col_a,
                'col_b':      layer.col_b,
                'confidence': layer.confidence,
                'coverage':   layer.coverage_frac,
                'grid':       grid,
                'row_indices': layer.row_indices
            })

        # Residual always gets full grid
        self.residual_grid_ = dict(self.base_grid)

        return self

    def pruned_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return X with validated-redundant features removed."""
        keep = [c for c in X.columns if c not in self.features_to_remove_]
        return X[keep]

    def global_grid(self) -> dict:
        """
        Compute the intersection of all layer grids — a single
        conservative pruned grid valid across the whole dataset.
        Useful when partition-level training is not desired.
        """
        if not self.layer_grids_:
            return dict(self.base_grid)

        result = dict(self.layer_grids_[0]['grid'])
        for lg in self.layer_grids_[1:]:
            for k, v in lg['grid'].items():
                if k in result:
                    merged = [x for x in v if x in result[k]]
                    if merged:
                        result[k] = merged
        return result

    def savings_estimate(self) -> dict:
        """Report estimated config reduction."""
        def count(g):
            t = 1
            for v in g.values():
                t *= len(v)
            return t

        base   = count(self.base_grid)
        global_= count(self.global_grid())
        return {
            'configs_base':   base,
            'configs_global': global_,
            'savings_pct':    round((base - global_) / base * 100, 1),
            'features_removed': len(self.features_to_remove_)
        }
