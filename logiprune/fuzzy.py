"""
logiprune.fuzzy
───────────────
Fuzzy disjunction (t-conorm) extension for LogiPrune.

When classical propositional analysis finds A∨B as the dominant
relation for a feature pair, it recovers structural information
(the pair is never jointly inactive) but provides no direct
pruning value — because A∨B in Boolean logic carries no
directional information about C.

This module extends A∨B to the continuous domain using t-conorms,
which are the standard mathematical generalization of OR to [0,1].

Four t-conorms are evaluated per pair:

    max(a,b)          — Zadeh maximum (classical fuzzy OR)
    a + b − a·b       — Probabilistic sum ("at least one fires")
    min(1, a+b)       — Bounded sum (Łukasiewicz t-conorm)
    (a+b)/2           — Arithmetic mean (not a true t-conorm,
                        but semantically "average activation")

For each t-conorm, the synthetic feature s = tconorm(A, B) is
analyzed against target C using SWTS. The best t-conorm is the
one whose synthetic feature has the highest propositional
confidence with C.

If that confidence ≥ threshold, the pair (A, B) is replaced by
the single synthetic feature s — genuine 2→1 feature compression
with preserved predictive information.

This directly addresses the information loss in:
    Boolean:    OR(0.25, 0.16) = 1       ← loses gradation
    Bounded:    min(1, 0.25+0.16) = 0.41  ← preserves magnitude
    Mean:       (0.25+0.16)/2  = 0.205   ← preserves average
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from .sweeper import SWTSSweeper
from .relations import A_IMPLIES_B, B_IMPLIES_A, BICONDITIONAL, CONTINGENCY


# ── T-conorm definitions ─────────────────────────────────────────────────────

TCONORMS = {
    'max':   lambda a, b: np.maximum(a, b),
    'prob':  lambda a, b: a + b - a * b,
    'bound': lambda a, b: np.minimum(1.0, a + b),
    'mean':  lambda a, b: (a + b) / 2.0,
}


@dataclass
class FuzzyResult:
    col_a:        str
    col_b:        str
    tconorm_name: str
    synth_name:   str          # name of the synthetic feature column
    confidence:   float        # of synth → target
    relation:     str          # propositional relation of synth with target
    threshold:    float        # T* for synth→target
    replaces:     tuple        # (col_a, col_b) — these are removed

    def __repr__(self):
        return (f"FuzzyResult({self.col_a} ∨[{self.tconorm_name}] {self.col_b} "
                f"→ '{self.synth_name}' | conf={self.confidence:.3f} "
                f"rel={self.relation})")


class FuzzyDisjunctionAnalyzer:
    """
    For pairs where A∨B is the dominant classical relation,
    computes the best t-conorm synthetic feature and tests
    whether it has a useful propositional relation with target C.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence for synth→C to accept compression (default 0.80).
    target_col : str
        Name of target column in the normalized+target DataFrame.
    """

    def __init__(self,
                 min_confidence: float = 0.80):
        self.min_confidence = min_confidence
        self.results_: list[FuzzyResult] = []
        self._synth_features: dict[str, np.ndarray] = {}

    def _best_tconorm(self,
                      a_vals: np.ndarray,
                      b_vals: np.ndarray,
                      c_vals: np.ndarray,
                      col_a: str,
                      col_b: str) -> Optional[FuzzyResult]:
        """
        Try all four t-conorms. Return the FuzzyResult with
        the highest propositional confidence against C, or None
        if no t-conorm achieves min_confidence.
        """
        best_conf   = -1.0
        best_result = None

        for name, fn in TCONORMS.items():
            synth = fn(a_vals, b_vals)

            # Run single-column SWTS: synth → C
            # Pack into a 2-column DataFrame for the sweeper
            df_pair = pd.DataFrame({
                '__synth__': synth,
                '__target__': c_vals
            })
            sweeper = SWTSSweeper(min_confidence=0.0)  # no filter; we filter below
            sweeper.fit(df_pair, col_pairs=[('__synth__', '__target__')])

            pair_result = sweeper.results_.get(('__synth__', '__target__'))
            if pair_result is None:
                continue

            if (pair_result.is_structural
                    and pair_result.confidence > best_conf
                    and pair_result.confidence >= self.min_confidence):
                best_conf = pair_result.confidence
                synth_name = f"{col_a}_OR_{name}_{col_b}"
                best_result = FuzzyResult(
                    col_a        = col_a,
                    col_b        = col_b,
                    tconorm_name = name,
                    synth_name   = synth_name,
                    confidence   = pair_result.confidence,
                    relation     = pair_result.relation,
                    threshold    = pair_result.threshold,
                    replaces     = (col_a, col_b)
                )
                self._synth_features[synth_name] = synth

        return best_result

    def analyze(self,
                X_norm: pd.DataFrame,
                y: pd.Series,
                aorb_pairs: list[tuple[str, str]]) -> list[FuzzyResult]:
        """
        Parameters
        ----------
        X_norm : normalized feature DataFrame (values in [0,1])
        y      : target series (will be normalized to [0,1] internally)
        aorb_pairs : list of (col_a, col_b) pairs where A∨B was found

        Returns list of FuzzyResult for pairs where compression is valid.
        """
        self.results_ = []

        # Normalize target to [0,1] for SWTS
        y_arr = y.values.astype(float)
        y_min, y_max = y_arr.min(), y_arr.max()
        c_norm = (y_arr - y_min) / (y_max - y_min + 1e-9)

        for col_a, col_b in aorb_pairs:
            if col_a not in X_norm.columns or col_b not in X_norm.columns:
                continue
            a_vals = X_norm[col_a].values
            b_vals = X_norm[col_b].values

            result = self._best_tconorm(a_vals, b_vals, c_norm, col_a, col_b)
            if result is not None:
                self.results_.append(result)

        return self.results_

    def apply_compression(self,
                          X: pd.DataFrame,
                          X_norm: pd.DataFrame,
                          y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Build compressed DataFrames: replace (col_a, col_b) pairs
        with their synthetic t-conorm features where valid.

        Returns
        -------
        X_compressed      : raw-scale DataFrame with synthetic cols added
        X_norm_compressed : normalized DataFrame with synthetic cols added
        removed_cols      : list of original columns that were replaced
        """
        removed = set()
        X_out      = X.copy()
        X_norm_out = X_norm.copy()

        # Normalize y for synth feature computation
        y_arr = y.values.astype(float)
        y_min, y_max = y_arr.min(), y_arr.max()
        c_norm = (y_arr - y_min) / (y_max - y_min + 1e-9)

        for res in self.results_:
            ca, cb = res.col_a, res.col_b
            if ca in removed or cb in removed:
                continue  # already consumed by another compression

            fn = TCONORMS[res.tconorm_name]
            a_norm = X_norm_out[ca].values
            b_norm = X_norm_out[cb].values
            synth_norm = fn(a_norm, b_norm)

            # Add synthetic feature to both DataFrames
            X_norm_out[res.synth_name] = synth_norm
            # For raw X: use same t-conorm on standardized values
            # (preserves relative magnitude for downstream scaler)
            a_raw = X_out[ca].values.astype(float)
            b_raw = X_out[cb].values.astype(float)
            X_out[res.synth_name] = fn(a_raw, b_raw)

            removed.add(ca)
            removed.add(cb)

        # Drop replaced columns
        X_out      = X_out.drop(columns=list(removed), errors='ignore')
        X_norm_out = X_norm_out.drop(columns=list(removed), errors='ignore')

        return X_out, X_norm_out, list(removed)

    def summary(self) -> pd.DataFrame:
        if not self.results_:
            return pd.DataFrame()
        return pd.DataFrame([{
            'col_a':     r.col_a,
            'col_b':     r.col_b,
            'tconorm':   r.tconorm_name,
            'synth':     r.synth_name,
            'conf_w_C':  round(r.confidence, 4),
            'relation':  r.relation,
        } for r in self.results_])
