"""
examples/breast_cancer_demo.py
───────────────────────────────
Quick demonstration of LogiPrune on the breast_cancer dataset.
Run with: python examples/breast_cancer_demo.py
"""

import time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from logiprune import LogiPrune

# ── Data ──────────────────────────────────────────────────────────────────────
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# ── Hyperparameter grid ───────────────────────────────────────────────────────
base_grid = {
    'svc__C':      [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly'],
    'svc__gamma':  ['scale', 'auto', 0.01, 0.1],
}

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

# ── Baseline GridSearch ───────────────────────────────────────────────────────
print("Running baseline GridSearch (48 configurations)...")
t0 = time.time()
gs_base = GridSearchCV(pipe, base_grid, cv=3, scoring='accuracy', n_jobs=-1)
gs_base.fit(X_train, y_train)
t_base = time.time() - t0
acc_base = gs_base.score(X_test, y_test)
print(f"  Time:     {t_base:.2f}s")
print(f"  Accuracy: {acc_base:.4f}")

# ── LogiPrune + GridSearch ────────────────────────────────────────────────────
print("\nRunning LogiPrune...")
lp = LogiPrune(base_grid=base_grid, verbose=True)
lp.fit(X_train, y_train)
print(lp.report())

X_train_p = lp.transform(X_train)
X_test_p  = lp.transform(X_test)
pruned    = lp.pruned_grid()

print(f"\nRunning GridSearch on pruned space ({lp.n_configs_pruned_} configurations)...")
t0 = time.time()
gs_lp = GridSearchCV(pipe, pruned, cv=3, scoring='accuracy', n_jobs=-1)
gs_lp.fit(X_train_p, y_train)
t_lp  = time.time() - t0 + lp.preprocessing_time_
acc_lp = gs_lp.score(X_test_p, y_test)

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  RESULTS")
print("="*50)
print(f"  Configs:  {lp.n_configs_base_} → {lp.n_configs_pruned_}  "
      f"({lp.savings_summary()['config_savings_pct']}% reduction)")
print(f"  Time:     {t_base:.2f}s → {t_lp:.2f}s  "
      f"({(t_base-t_lp)/t_base*100:.1f}% reduction)")
print(f"  Accuracy: {acc_base:.4f} → {acc_lp:.4f}  "
      f"(Δ={acc_lp-acc_base:+.4f})")
print("="*50)
