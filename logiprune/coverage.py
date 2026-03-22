"""
logiprune.coverage
──────────────────
Iterative Propositional Coverage (IPC).

This is the core v3 innovation. The key insight:

    A dataset D does not have ONE logical structure.
    It has a FAMILY of local structures, each valid over
    a subset of observations.

Algorithm:
──────────
1. Run SWTS on all pairs over full D → find strongest relation R₁
   covering subset D₁ ⊆ D (the observations that satisfy R₁).
2. Set residual D' = D \ D₁.
3. Run SWTS on D' → find R₂ covering D₂ ⊆ D'.
4. Repeat until:
   - Residual < min_residual_frac × |D|, OR
   - Best confidence in residual < min_confidence, OR
   - max_iterations reached.

Output: a list of LogicLayer objects, each with:
  - the relation found
  - the row indices it covers
  - the features involved
  - the threshold used
  - the grid restrictions it implies

This enables PARTITION-LEVEL GridSearch:
  - Each layer gets its own pruned grid
  - Models are trained on the covered subset
  - Final prediction = ensemble by layer membership

Cost control:
  Total SWTS cost across all iterations =
  O(n × C² × S × k) where k = iterations.
  With n=10000, C=15, S=16, k=5: ~1.8×10⁸ ops.
  GridSearch over 48 configs × 3-fold CV on same data:
  O(48 × 3 × n × features²) ≈ O(48 × 3 × 10000 × 225) = 3.24×10⁹ ops.
  → IPC overhead is ~5-6% of GridSearch cost.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from .sweeper import SWTSSweeper
from .discretize import AdaptiveDiscretizer
from .relations import (BICONDITIONAL, A_IMPLIES_B, B_IMPLIES_A,
                        INCOMPATIBLE, A_OR_B, CONTINGENCY)


@dataclass
class LogicLayer:
    """A propositional structure valid over a subset of observations."""
    layer_id:      int
    relation:      str
    col_a:         str
    col_b:         str
    confidence:    float
    support:       float
    threshold:     float
    row_indices:   np.ndarray        # indices into original dataset
    coverage_frac: float             # fraction of total dataset
    grid_restrictions: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"Layer {self.layer_id}: {self.col_a} {self.relation} {self.col_b} "
                f"| conf={self.confidence:.3f} | covers {self.coverage_frac:.1%} "
                f"| n={len(self.row_indices)}")


class IterativeCoverage:
    """
    Builds a layered propositional map of the dataset.

    Parameters
    ----------
    min_confidence : float
        Stop if best relation confidence falls below this (default 0.75).
    min_residual_frac : float
        Stop if residual is smaller than this fraction of total (default 0.05).
    max_iterations : int
        Hard stop on number of layers (default 8).
    coverage_threshold : float
        Fraction of observations that must satisfy a relation for it to
        count as 'covered' (default 0.60).
    acc_drop_tolerance : float
        Max allowed accuracy drop from feature elimination (default 0.04).
    discretizer_strategy : str
        Strategy for AdaptiveDiscretizer (default 'percentile').
    """

    def __init__(self,
                 min_confidence: float = 0.75,
                 min_residual_frac: float = 0.05,
                 max_iterations: int = 8,
                 coverage_threshold: float = 0.60,
                 acc_drop_tolerance: float = 0.04,
                 discretizer_strategy: str = 'percentile'):
        self.min_confidence      = min_confidence
        self.min_residual_frac   = min_residual_frac
        self.max_iterations      = max_iterations
        self.coverage_threshold  = coverage_threshold
        self.acc_drop_tolerance  = acc_drop_tolerance
        self.discretizer_strategy = discretizer_strategy

        self.layers_: list[LogicLayer] = []
        self.uncovered_indices_: np.ndarray = np.array([])
        self.discretizer_: Optional[AdaptiveDiscretizer] = None
        self._n_total: int = 0

    # ── Grid restriction rules ───────────────────────────────────────────────

    def _grid_restrictions(self, relation: str, confidence: float) -> dict:
        """
        Translate a propositional relation into GridSearch constraints.

        Rules:
          BICONDITIONAL (A↔B):
            → Eliminate one feature. Linear kernel sufficient.
          A→B or B→A, conf ≥ 0.95:
            → Strong linear signal. kernel=linear, C compressed.
          A→B or B→A, conf ∈ [0.85, 0.95):
            → Dominant linear signal. Exclude rbf.
          INCOMPATIBLE (A→¬B):
            → Features activate in separate contexts. Keep both but
              restrict to linear/poly (interaction kernel misleads).
          A∨B:
            → Co-activation structure. No strong restriction.
          CONTINGENCY:
            → No restriction. Full grid required.
        """
        if relation == BICONDITIONAL:
            return {'action': 'eliminate_one',
                    'svc__kernel': ['linear'],
                    'svc__gamma': ['scale'],
                    'svc__C': [0.1, 1, 10]}

        elif relation in (A_IMPLIES_B, B_IMPLIES_A):
            if confidence >= 0.95:
                return {'action': 'restrict_kernel',
                        'svc__kernel': ['linear'],
                        'svc__gamma': ['scale'],
                        'svc__C': [0.1, 1, 10]}
            else:
                return {'action': 'restrict_kernel',
                        'svc__kernel': ['linear', 'poly'],
                        'svc__C': [0.1, 1, 10, 100]}

        elif relation == INCOMPATIBLE:
            return {'action': 'restrict_kernel',
                    'svc__kernel': ['linear', 'poly'],
                    'svc__C': [0.1, 1, 10, 100]}

        elif relation == A_OR_B:
            return {'action': 'none',
                    'note': 'co-activation structure; minor restriction only',
                    'svc__C': [0.1, 1, 10, 100]}

        else:  # CONTINGENCY
            return {'action': 'none'}

    # ── Row coverage ────────────────────────────────────────────────────────

    def _covered_rows(self,
                      X_norm: pd.DataFrame,
                      col_a: str, col_b: str,
                      relation: str,
                      threshold: float) -> np.ndarray:
        """
        Return boolean mask of rows satisfying the relation at threshold.
        """
        a = (X_norm[col_a].values > threshold).astype(int)
        b = (X_norm[col_b].values > threshold).astype(int)

        if relation == BICONDITIONAL:
            mask = (a == b)                          # A=B (both 0 or both 1)
        elif relation == A_IMPLIES_B:
            mask = ~((a == 1) & (b == 0))            # NOT (A=1 AND B=0)
        elif relation == B_IMPLIES_A:
            mask = ~((a == 0) & (b == 1))            # NOT (A=0 AND B=1)
        elif relation == INCOMPATIBLE:
            mask = ~((a == 1) & (b == 1))            # NOT (A=1 AND B=1)
        elif relation == A_OR_B:
            mask = (a == 1) | (b == 1)               # at least one active
        else:
            mask = np.ones(len(a), dtype=bool)       # contingency: all rows

        return mask

    # ── Main fit ─────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'IterativeCoverage':
        self._n_total = len(X)
        self.layers_ = []

        # Discretize full dataset once
        self.discretizer_ = AdaptiveDiscretizer(strategy=self.discretizer_strategy)
        X_norm_full = self.discretizer_.fit_transform(X)

        residual_idx = np.arange(len(X))   # all row indices initially

        for iteration in range(self.max_iterations):
            n_residual = len(residual_idx)

            # Stop if residual too small
            if n_residual / self._n_total < self.min_residual_frac:
                break

            X_res = X_norm_full.iloc[residual_idx].reset_index(drop=True)

            # Run SWTS on residual
            sweeper = SWTSSweeper(min_confidence=self.min_confidence)
            sweeper.fit(X_res)

            structural = sweeper.structural_pairs(min_confidence=self.min_confidence)

            if not structural:
                break  # No more structure to find

            # Pick best pair by confidence × support
            structural.sort(
                key=lambda x: x[1].confidence * x[1].support
                              * (4.0 * x[1].threshold * (1.0 - x[1].threshold)),
                reverse=True)
            (best_ca, best_cb), best_r = structural[0]

            # Stop if best confidence too low
            if best_r.confidence < self.min_confidence:
                break

            # Find covered rows within residual
            covered_mask = self._covered_rows(
                X_res, best_ca, best_cb, best_r.relation, best_r.threshold)
            coverage_frac_local = covered_mask.mean()

            # Only accept layer if it covers enough of the residual
            if coverage_frac_local < self.coverage_threshold:
                break

            covered_global_idx = residual_idx[covered_mask]
            coverage_frac_total = len(covered_global_idx) / self._n_total

            grid_restr = self._grid_restrictions(best_r.relation, best_r.confidence)

            layer = LogicLayer(
                layer_id      = iteration,
                relation      = best_r.relation,
                col_a         = best_ca,
                col_b         = best_cb,
                confidence    = best_r.confidence,
                support       = best_r.support,
                threshold     = best_r.threshold,
                row_indices   = covered_global_idx,
                coverage_frac = coverage_frac_total,
                grid_restrictions = grid_restr
            )
            self.layers_.append(layer)

            # Residual = rows NOT covered by this layer
            residual_idx = residual_idx[~covered_mask]

        self.uncovered_indices_ = residual_idx
        return self

    # ── Reporting ─────────────────────────────────────────────────────────────

    def coverage_report(self) -> pd.DataFrame:
        rows = []
        for l in self.layers_:
            rows.append(dict(
                layer    = l.layer_id,
                col_a    = l.col_a,
                col_b    = l.col_b,
                relation = l.relation,
                conf     = round(l.confidence, 4),
                support  = round(l.support, 4),
                thresh   = round(l.threshold, 3),
                n_obs    = len(l.row_indices),
                pct_total= round(l.coverage_frac * 100, 1),
                grid_action = l.grid_restrictions.get('action', 'none')
            ))
        uncov_pct = round(len(self.uncovered_indices_) / self._n_total * 100, 1)
        rows.append(dict(layer='uncovered', col_a='—', col_b='—',
                         relation='contingency', conf=0, support=0,
                         thresh=0, n_obs=len(self.uncovered_indices_),
                         pct_total=uncov_pct, grid_action='full_grid'))
        return pd.DataFrame(rows)

    @property
    def total_covered_frac(self) -> float:
        return sum(l.coverage_frac for l in self.layers_)
