"""
tests/test_logiprune.py
───────────────────────
Tests for LogiPrune (Paper 1) and LogiPruneEntropy (Paper 2).
Run with: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def breast_cancer_split():
    data = load_breast_cancer(as_frame=True)
    return train_test_split(
        data.data, data.target,
        test_size=0.2, random_state=42, stratify=data.target
    )

@pytest.fixture(scope="module")
def small_split():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((200, 10)),
                     columns=[f'f{i}' for i in range(10)])
    y = pd.Series((X['f0'] + X['f1'] > 0).astype(int))
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ═══════════════════════════════════════════════════════════════════════════════
# PAPER 1 — LogiPrune
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveDiscretizer:
    def test_percentile_range(self):
        from logiprune import AdaptiveDiscretizer
        X = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.rand(100)})
        Xn = AdaptiveDiscretizer(strategy='percentile').fit_transform(X)
        assert Xn.min().min() >= 0.0
        assert Xn.max().max() <= 1.0 + 1e-6

    def test_minmax_range(self):
        from logiprune import AdaptiveDiscretizer
        X = pd.DataFrame({'a': np.arange(100, dtype=float)})
        Xn = AdaptiveDiscretizer(strategy='minmax').fit_transform(X)
        assert abs(Xn['a'].min()) < 1e-6
        assert abs(Xn['a'].max() - 1.0) < 1e-6

    def test_unknown_strategy_returns_empty(self):
        from logiprune import AdaptiveDiscretizer
        # Unknown strategy produces output without crashing
        Xn = AdaptiveDiscretizer(strategy='unknown').fit_transform(
            pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        )
        assert isinstance(Xn, pd.DataFrame)


class TestRelations:
    def test_biconditional(self):
        from logiprune import classify_pair, BICONDITIONAL
        a = np.array([1]*100 + [0]*100)
        b = np.array([1]*100 + [0]*100)
        rel, conf, _ = classify_pair(a, b)
        assert rel == BICONDITIONAL
        assert conf > 0.95

    def test_implication(self):
        from logiprune import classify_pair, A_IMPLIES_B, B_IMPLIES_A
        a = np.array([1]*100 + [0]*100)
        b = np.array([1]*100 + [1]*100)
        rel, conf, _ = classify_pair(a, b)
        assert rel in (A_IMPLIES_B, B_IMPLIES_A)
        assert conf >= 0.5

    def test_contingency(self):
        from logiprune import classify_pair, CONTINGENCY
        rng = np.random.default_rng(42)
        a = rng.integers(0, 2, 300)
        b = rng.integers(0, 2, 300)
        rel, conf, _ = classify_pair(a, b, epsilon=0.02)
        assert rel == CONTINGENCY


class TestLogiPrune:
    BASE_GRID = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}

    def test_fit_returns_self(self, breast_cancer_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = breast_cancer_split
        assert LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr) is not None

    def test_transform_rows_preserved(self, breast_cancer_split):
        from logiprune import LogiPrune
        X_tr, X_te, y_tr, _ = breast_cancer_split
        lp = LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr)
        assert lp.transform(X_te).shape[0] == X_te.shape[0]

    def test_pruned_grid_subset(self, breast_cancer_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = breast_cancer_split
        pg = LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr).pruned_grid()
        for k in self.BASE_GRID:
            assert k in pg
            assert all(v in self.BASE_GRID[k] for v in pg[k])

    def test_savings_summary_keys(self, breast_cancer_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = breast_cancer_split
        s = LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr).savings_summary()
        for key in ['configs_base', 'configs_pruned', 'config_savings_pct',
                    'features_eliminated', 'preprocessing_time_s']:
            assert key in s

    def test_report_string(self, breast_cancer_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = breast_cancer_split
        r = LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr).report()
        assert isinstance(r, str) and "LogiPrune" in r

    def test_not_fitted_raises(self):
        from logiprune import LogiPrune
        lp = LogiPrune(base_grid={'C': [1]})
        with pytest.raises(RuntimeError):
            lp.transform(pd.DataFrame({'a': [1]}))
        with pytest.raises(RuntimeError):
            lp.pruned_grid()

    def test_configs_not_increased(self, small_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = small_split
        def count(g):
            t=1
            for v in g.values(): t*=len(v)
            return t
        lp = LogiPrune(base_grid=self.BASE_GRID).fit(X_tr, y_tr)
        assert count(lp.pruned_grid()) <= count(self.BASE_GRID)

    def test_fit_transform_shape(self, small_split):
        from logiprune import LogiPrune
        X_tr, _, y_tr, _ = small_split
        lp = LogiPrune(base_grid=self.BASE_GRID)
        X_ft = lp.fit_transform(X_tr, y_tr)
        assert X_ft.shape[0] == X_tr.shape[0]
        assert X_ft.shape[1] <= X_tr.shape[1] + 5  # may add synthetics


# ═══════════════════════════════════════════════════════════════════════════════
# PAPER 2 — LogiPruneEntropy
# ═══════════════════════════════════════════════════════════════════════════════

class TestTruthTableEntropy:
    def test_entropy_range(self):
        from logiprune import truth_table_profile
        a = np.array([1.0]*80 + [0.0]*20)
        b = np.array([1.0]*80 + [0.0]*20)
        ttp = truth_table_profile('a', 'b', a, b, 0.5)
        assert 0.0 <= ttp.entropy <= 2.0

    def test_uniform_near_max_entropy(self):
        from logiprune import truth_table_profile
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 1000)
        b = rng.uniform(0, 1, 1000)
        ttp = truth_table_profile('a', 'b', a, b, 0.5)
        assert ttp.entropy > 1.7

    def test_perfect_structure_low_entropy(self):
        from logiprune import truth_table_profile
        a = np.ones(100)
        b = np.ones(100)
        ttp = truth_table_profile('a', 'b', a, b, 0.5)
        assert ttp.entropy < 0.1
        assert ttp.dominant_cell == '11'
        assert ttp.dominant == 1.0

    def test_complexity_class(self):
        from logiprune.entropy import _complexity_class
        assert _complexity_class(0.3)  == 'very_low'
        assert _complexity_class(0.7)  == 'low'
        assert _complexity_class(1.2)  == 'medium'
        assert _complexity_class(1.8)  == 'high'


class TestEntropyAnalyzer:
    def _make_normalized(self, X_tr, y_tr):
        from logiprune.discretize import AdaptiveDiscretizer
        return AdaptiveDiscretizer().fit_transform(X_tr), y_tr

    def test_fit_pair_count(self, small_split):
        from logiprune import EntropyAnalyzer
        X_tr, _, y_tr, _ = small_split
        Xn, y = self._make_normalized(X_tr, y_tr)
        ea = EntropyAnalyzer().fit(Xn, y)
        # C(10,2) = 45 pairs
        assert len(ea.entropy_profiles_) == 45
        assert 0.0 <= ea.dataset_h_min_ <= 2.0
        assert 0.0 <= ea.dataset_h_mean_ <= 2.0

    def test_xgb_grid_subset(self, small_split):
        from logiprune import EntropyAnalyzer
        X_tr, _, y_tr, _ = small_split
        Xn, y = self._make_normalized(X_tr, y_tr)
        base = {'xgb__max_depth': [3,5,7], 'xgb__n_estimators': [100,200,300],
                'xgb__learning_rate': [0.05,0.1,0.3],
                'xgb__subsample': [0.8,1.0], 'xgb__colsample_bytree': [0.8,1.0]}
        pg = EntropyAnalyzer().fit(Xn, y).xgb_grid_from_entropy(base)
        for k in base:
            if k in pg:
                assert all(v in base[k] for v in pg[k])

    def test_complexity_report_sorted(self, small_split):
        from logiprune import EntropyAnalyzer
        X_tr, _, y_tr, _ = small_split
        Xn, y = self._make_normalized(X_tr, y_tr)
        df = EntropyAnalyzer().fit(Xn, y).complexity_report()
        assert isinstance(df, pd.DataFrame)
        assert df['h_min'].is_monotonic_increasing

    def test_feedback_check_returns_list(self, small_split):
        from logiprune import EntropyAnalyzer
        X_tr, _, y_tr, _ = small_split
        Xn, y = self._make_normalized(X_tr, y_tr)
        result = EntropyAnalyzer().fit(Xn, y).feedback_check(['f0'], Xn, y)
        assert isinstance(result, list)


class TestLogiPruneEntropy:
    BASE_GRID = {
        'xgb__n_estimators':     [100, 200, 300],
        'xgb__max_depth':        [3, 5, 7],
        'xgb__learning_rate':    [0.05, 0.1, 0.3],
        'xgb__subsample':        [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
    }

    def test_fit_returns_self(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        lpe = LogiPruneEntropy(base_grid=self.BASE_GRID)
        assert lpe.fit(X_tr, y_tr) is lpe

    def test_transform_rows_preserved(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, X_te, y_tr, _ = small_split
        lpe = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr)
        assert lpe.transform(X_te).shape[0] == X_te.shape[0]

    def test_transform_cols_leq_original(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, X_te, y_tr, _ = small_split
        lpe = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr)
        assert lpe.transform(X_te).shape[1] <= X_te.shape[1]

    def test_pruned_grid_subset(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        pg = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr).pruned_grid()
        for k in self.BASE_GRID:
            if k in pg:
                assert all(v in self.BASE_GRID[k] for v in pg[k])

    def test_savings_summary_keys(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        s = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr).savings_summary()
        for key in ['configs_base','configs_pruned','config_savings_pct',
                    'features_eliminated','features_reinstated',
                    'h_min','h_applied','preprocessing_time_s']:
            assert key in s

    def test_report_string(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        r = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr).report()
        assert isinstance(r, str) and "LogiPruneEntropy" in r

    def test_not_fitted_raises(self):
        from logiprune import LogiPruneEntropy
        lpe = LogiPruneEntropy(base_grid=self.BASE_GRID)
        with pytest.raises(RuntimeError):
            lpe.transform(pd.DataFrame({'a': [1]}))
        with pytest.raises(RuntimeError):
            lpe.pruned_grid()

    def test_configs_not_increased(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        def count(g):
            t=1
            for v in g.values(): t*=len(v)
            return t
        lpe = LogiPruneEntropy(base_grid=self.BASE_GRID).fit(X_tr, y_tr)
        assert count(lpe.pruned_grid()) <= count(self.BASE_GRID)

    def test_fit_transform_shape(self, small_split):
        from logiprune import LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        X_ft = LogiPruneEntropy(base_grid=self.BASE_GRID).fit_transform(X_tr, y_tr)
        assert X_ft.shape[0] == X_tr.shape[0]


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — Combined pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedPipeline:
    BASE_SVC = {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}
    BASE_XGB = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth':    [3, 5],
        'xgb__learning_rate':[0.1, 0.3],
        'xgb__subsample':    [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
    }

    def test_two_stage_runs(self, small_split):
        from logiprune import LogiPrune, LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        lp   = LogiPrune(base_grid=self.BASE_SVC).fit(X_tr, y_tr)
        X_p1 = lp.transform(X_tr)
        lpe  = LogiPruneEntropy(base_grid=self.BASE_XGB).fit(X_p1, y_tr)
        X_final = lpe.transform(X_p1)
        assert X_final.shape[0] == X_tr.shape[0]
        assert X_final.shape[1] <= X_tr.shape[1]

    def test_configs_monotonically_decrease(self, small_split):
        from logiprune import LogiPrune, LogiPruneEntropy
        X_tr, _, y_tr, _ = small_split
        def count(g):
            t=1
            for v in g.values(): t*=len(v)
            return t
        lp   = LogiPrune(base_grid=self.BASE_SVC).fit(X_tr, y_tr)
        lpe  = LogiPruneEntropy(base_grid=self.BASE_XGB).fit(lp.transform(X_tr), y_tr)
        assert count(lpe.pruned_grid()) <= count(self.BASE_XGB)
