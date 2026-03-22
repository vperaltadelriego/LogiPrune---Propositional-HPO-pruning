"""
logiprune.sweeper
─────────────────
Stability-Weighted Threshold Sweeping (SWTS).

For a pair of continuous columns, finds T* = argmax S(T_h) where:

    S(T_h) = confidence(T_h) × support(T_h) × stability(T_h)

    stability(T_h) = 1 - |conf(T_h+δ) - conf(T_h-δ)| / 2δ

T* is the threshold at which the propositional relationship between
the two columns is simultaneously most confident, most supported,
and most resistant to perturbation.

Cost: O(C² × S × n) where C=columns, S=steps, n=rows.
With C=15, S=16, n=10000: ~3.6M ops — well below GridSearch cost.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from .relations import classify_pair, RelationResult, CONTINGENCY


class SWTSSweeper:
    """
    Runs SWTS over all feature pairs in a normalized DataFrame.

    Parameters
    ----------
    steps : array-like
        Threshold values to sweep. Default: 0.10 to 0.85 in steps of 0.05.
    epsilon : float
        Rare-tuple threshold for relation classification (default 0.05).
    delta : float
        Perturbation for stability computation (default 0.05).
    min_confidence : float
        Minimum confidence to consider a relation structural (default 0.70).
    max_cost_ratio : float
        If sweeping all pairs would cost > max_cost_ratio × n, subsample
        columns by MI with target. Safety valve for very wide datasets.
    """

    def __init__(self,
                 steps=None,
                 epsilon: float = 0.05,
                 delta: float = 0.05,
                 min_confidence: float = 0.70,
                 max_cost_ratio: float = 50.0):
        self.steps         = np.arange(0.25, 0.76, 0.05) if steps is None else np.array(steps)
        self.epsilon       = epsilon
        self.delta         = delta
        self.min_confidence = min_confidence
        self.max_cost_ratio = max_cost_ratio

        # Results: keyed by (col_a, col_b)
        self.results_: dict[Tuple[str,str], RelationResult] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _binarize(self, v: np.ndarray, th: float) -> np.ndarray:
        return (v > th).astype(np.int8)

    def _confidence_ab(self, a: np.ndarray, b: np.ndarray) -> float:
        m = a == 1
        return float((b[m] == 1).mean()) if m.sum() > 0 else 0.0

    def _stability(self, ca: np.ndarray, cb: np.ndarray, th: float) -> float:
        tp = min(th + self.delta, 0.90)
        tm = max(th - self.delta, 0.10)
        cp = self._confidence_ab(self._binarize(ca, tp), self._binarize(cb, tp))
        cm = self._confidence_ab(self._binarize(ca, tm), self._binarize(cb, tm))
        return 1.0 - abs(cp - cm) / (2 * self.delta + 1e-9)

    def _sweep_pair(self, ca: np.ndarray, cb: np.ndarray) -> Tuple[float, RelationResult]:
        """Return (best_score, RelationResult) for a single pair."""
        best_score = -1.0
        best_result = None

        for th in self.steps:
            a = self._binarize(ca, th)
            b = self._binarize(cb, th)

            rel, conf, supp = classify_pair(a, b, self.epsilon)
            stab     = self._stability(ca, cb, th)
            # Centrality penalty: 4·Th·(1-Th) → 0 at extremes, 1.0 at Th=0.5
            # Suppresses trivial relations at range boundaries.
            # Automatically adapts to any dataset scale because it operates
            # on the normalized [0,1] threshold value, not raw feature values.
            penalty  = 4.0 * th * (1.0 - th)
            score    = conf * supp * stab * penalty

            if score > best_score:
                best_score = score
                n = len(a)
                best_result = RelationResult(
                    relation   = rel,
                    confidence = conf,
                    support    = supp,
                    stability  = stab,
                    threshold  = th,
                    n11 = int(((a==1)&(b==1)).sum()),
                    n10 = int(((a==1)&(b==0)).sum()),
                    n01 = int(((a==0)&(b==1)).sum()),
                    n00 = int(((a==0)&(b==0)).sum()),
                    n_total = n
                )

        return best_score, best_result

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame,
            col_pairs: Optional[list] = None) -> 'SWTSSweeper':
        """
        Run SWTS over all pairs (or a specified subset).

        Parameters
        ----------
        X : DataFrame, values in [0, 1] (output of AdaptiveDiscretizer)
        col_pairs : list of (col_a, col_b) tuples, or None for all pairs
        """
        self.results_ = {}
        cols = X.columns.tolist()

        if col_pairs is None:
            col_pairs = [(cols[i], cols[j])
                         for i in range(len(cols))
                         for j in range(i+1, len(cols))]

        for ca, cb in col_pairs:
            _, result = self._sweep_pair(X[ca].values, X[cb].values)
            if result is not None:
                self.results_[(ca, cb)] = result

        return self

    def structural_pairs(self, min_confidence: Optional[float] = None) -> list:
        """Return pairs with structural (non-contingency) relations."""
        mc = min_confidence or self.min_confidence
        return [(pair, r) for pair, r in self.results_.items()
                if r.is_structural and r.confidence >= mc]

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame summarising all pair results."""
        rows = []
        for (ca, cb), r in self.results_.items():
            rows.append(dict(
                col_a=ca, col_b=cb,
                relation=r.relation,
                confidence=round(r.confidence, 4),
                support=round(r.support, 4),
                stability=round(r.stability, 4),
                threshold=round(r.threshold, 3),
                coverage=round(r.coverage, 4),
                n11=r.n11, n10=r.n10, n01=r.n01, n00=r.n00
            ))
        return pd.DataFrame(rows).sort_values('confidence', ascending=False)
