"""
logiprune.discretize
────────────────────
Adaptive discretization strategies for continuous features.

The key insight: standard min-max normalization to [0,1] flattens
density structure (e.g. ages 15-90 where 80% cluster at 20-50).
Percentile-based discretization preserves that density, making
threshold sweeping semantically meaningful.

Three strategies:
  - 'percentile'  : divide by deciles/quintiles of actual distribution
  - 'minmax'      : standard [0,1] normalization (baseline)
  - 'zscore_clip' : z-score clamped to [-3, 3] then normalized
"""

import numpy as np
import pandas as pd
from typing import Literal


DiscretizeStrategy = Literal['percentile', 'minmax', 'zscore_clip']


class AdaptiveDiscretizer:
    """
    Fit on training data, transform any split to [0, 1] using
    the chosen strategy. Percentile is the default and recommended
    strategy for LogiPrune.

    Parameters
    ----------
    strategy : 'percentile' | 'minmax' | 'zscore_clip'
    n_quantiles : int
        Number of quantile breakpoints (default 10 = deciles).
        Only used when strategy='percentile'.
    """

    def __init__(self,
                 strategy: DiscretizeStrategy = 'percentile',
                 n_quantiles: int = 10):
        self.strategy    = strategy
        self.n_quantiles = n_quantiles
        self._params: dict = {}   # col -> (low, high) or quantile edges

    def fit(self, X: pd.DataFrame) -> 'AdaptiveDiscretizer':
        self._params = {}
        for col in X.columns:
            v = X[col].values.astype(float)
            if self.strategy == 'percentile':
                edges = np.percentile(v, np.linspace(0, 100, self.n_quantiles + 1))
                # Deduplicate edges (can happen with discrete-like columns)
                edges = np.unique(edges)
                self._params[col] = edges
            elif self.strategy == 'minmax':
                self._params[col] = (v.min(), v.max())
            elif self.strategy == 'zscore_clip':
                self._params[col] = (v.mean(), v.std() + 1e-9)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in X.columns:
            v = X[col].values.astype(float)
            if self.strategy == 'percentile':
                edges = self._params[col]
                # Map each value to its quantile rank in [0, 1]
                out[col] = np.searchsorted(edges, v, side='right') / len(edges)
            elif self.strategy == 'minmax':
                lo, hi = self._params[col]
                out[col] = (v - lo) / (hi - lo + 1e-9)
            elif self.strategy == 'zscore_clip':
                mu, sd = self._params[col]
                z = (v - mu) / sd
                out[col] = (np.clip(z, -3, 3) + 3) / 6
        return pd.DataFrame(out, columns=X.columns)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
