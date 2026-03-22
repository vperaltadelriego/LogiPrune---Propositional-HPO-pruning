"""
examples/xgboost_entropy_demo.py
─────────────────────────────────
Demonstration of LogiPruneEntropy (Paper 2) on breast_cancer.

Shows:
  - Truth table entropy analysis
  - XGBoost grid restriction from entropy signal
  - Feedback loop (feature reinstatement check)
  - Comparison: Baseline GridSearch vs LogiPruneEntropy

Run with: python examples/xgboost_entropy_demo.py
"""

import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, recall_score

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("This example requires xgboost. Install with: pip install xgboost")

from logiprune import LogiPruneEntropy

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# ── XGBoost grid (108 configurations) ────────────────────────────────────────
base_grid = {
    'xgb__n_estimators':     [100, 200, 300],
    'xgb__max_depth':        [3, 5, 7],
    'xgb__learning_rate':    [0.05, 0.1, 0.3],
    'xgb__subsample':        [0.8, 1.0],
    'xgb__colsample_bytree': [0.8, 1.0],
}

def xgb_pipe():
    return Pipeline([('xgb', XGBClassifier(
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, verbosity=0))])

def evaluate(model, X_te, y_te):
    yp = model.predict(X_te)
    return dict(
        f1  =round(f1_score(y_te, yp),4),
        acc =round(accuracy_score(y_te, yp),4),
        rec =round(recall_score(y_te, yp),4),
    )

# ── Baseline GridSearch ───────────────────────────────────────────────────────
print("Running Baseline GridSearch (108 configurations)...")
t0 = time.time()
gs_base = GridSearchCV(xgb_pipe(), base_grid, cv=3, scoring='f1', n_jobs=-1)
gs_base.fit(X_train, y_train)
t_base = time.time() - t0
m_base = evaluate(gs_base, X_test, y_test)
print(f"  Time: {t_base:.1f}s  F1: {m_base['f1']}  Acc: {m_base['acc']}  Rec: {m_base['rec']}")

# ── LogiPruneEntropy ──────────────────────────────────────────────────────────
print("\nRunning LogiPruneEntropy...")
lpe = LogiPruneEntropy(base_grid=base_grid, verbose=True)
lpe.fit(X_train, y_train)
print(lpe.report())

X_train_e = lpe.transform(X_train)
X_test_e  = lpe.transform(X_test)
pruned    = lpe.pruned_grid()

print(f"\nRunning GridSearch on entropy-pruned space...")
t0 = time.time()
gs_lpe = GridSearchCV(xgb_pipe(), pruned, cv=3, scoring='f1', n_jobs=-1)
gs_lpe.fit(X_train_e, y_train)
t_lpe = time.time() - t0 + lpe.preprocessing_time_
m_lpe = evaluate(gs_lpe, X_test_e, y_test)

# ── Results ───────────────────────────────────────────────────────────────────
s = lpe.savings_summary()
print("\n" + "="*55)
print("  RESULTS")
print("="*55)
print(f"  H* (min entropy):     {s['h_min']}")
print(f"  H_applied (weighted): {s['h_applied']}")
print(f"  Configs: {s['configs_base']} → {s['configs_pruned']} "
      f"({s['config_savings_pct']}% reduction)")
print(f"  Features eliminated:  {s['features_eliminated']}")
print(f"  Features reinstated:  {s['features_reinstated']}")
print()
print(f"  {'Method':<20} {'Time':>7}  {'F1':>7}  {'Acc':>7}  {'Rec':>7}")
print(f"  {'-'*52}")
print(f"  {'Baseline (108 cfgs)':<20} {t_base:>6.1f}s  "
      f"{m_base['f1']:>7}  {m_base['acc']:>7}  {m_base['rec']:>7}")
print(f"  {'LogiPruneEntropy':<20} {t_lpe:>6.1f}s  "
      f"{m_lpe['f1']:>7}  {m_lpe['acc']:>7}  {m_lpe['rec']:>7}")
print()
df1 = round(m_lpe['f1'] - m_base['f1'], 4)
dt  = round((t_base - t_lpe) / t_base * 100, 1)
print(f"  ΔF1:  {df1:+.4f}   ΔTime: {dt:+.1f}%")
print("="*55)
