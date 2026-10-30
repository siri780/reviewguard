"""
ReviewGuard - M6 SHAP Explainability (corrected)

Fix applied after review: this script previously recomputed its own degree/
PageRank features independently (one-way `row<col` edges, PageRank on
mat["homo"]) and used its own separate `train_test_split(random_state=42)`
- so SHAP was explaining a DIFFERENT feature set and split than the
corrected M3 model, not the one actually being reported. It now loads the
exact feature table M3 wrote (results/_m3_features.csv: bidirectional
PageRank scoped to R-U-R+R-T-R) and the exact shared split
(results/_shared_split_row_ids.csv), so the SHAP explanation corresponds to
the model whose metrics are in m3_network_results.txt.

Run this AFTER m3_network_features.py.

Reports:
  - Fidelity: SHAP values exactly reconstruct the model's linear margin (exact for linear models)
  - Top features by mean |SHAP|
  - Stability: top-5 feature ranking consistency across 3 bootstrap resamples
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

import common

merged_pdf = pd.read_csv(common.M3_FEATURES_PATH)
train_pdf, test_pdf = common.stratified_split(merged_pdf, reuse=True)

handcrafted_cols = [f"f{i}" for i in range(32)]
feature_cols = handcrafted_cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]

X_train, y_train = train_pdf[feature_cols].values, train_pdf["label"].values
X_test, y_test = test_pdf[feature_cols].values, test_pdf["label"].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
test_acc = clf.score(X_test_s, y_test)
print(f"sklearn LR test accuracy: {test_acc:.4f} (same shared split as M1/M3; NOT the headline metric)")

explainer = shap.LinearExplainer(clf, X_train_s)
shap_values = explainer.shap_values(X_test_s)

margin_from_model = clf.decision_function(X_test_s)
margin_from_shap = shap_values.sum(axis=1) + explainer.expected_value
fidelity_max_err = np.max(np.abs(margin_from_model - margin_from_shap))
print(f"Fidelity check - max abs error (SHAP sum vs model margin): {fidelity_max_err:.8f} (expect ~0)")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:10]
print("\nTop 10 features by mean |SHAP value|:")
top_features_report = []
for i in top_idx:
    line = f"  {feature_cols[i]}: {mean_abs_shap[i]:.4f}"
    print(line)
    top_features_report.append(line)

rankings = []
rng = np.random.RandomState(0)
for b in range(3):
    idx = rng.choice(len(X_train_s), size=len(X_train_s), replace=True)
    clf_b = LogisticRegression(max_iter=1000)
    clf_b.fit(X_train_s[idx], y_train[idx])
    exp_b = shap.LinearExplainer(clf_b, X_train_s[idx])
    sv_b = exp_b.shap_values(X_test_s)
    mean_abs_b = np.abs(sv_b).mean(axis=0)
    top5_b = set(np.argsort(mean_abs_b)[::-1][:5])
    rankings.append(top5_b)

overlap_01 = len(rankings[0] & rankings[1])
overlap_02 = len(rankings[0] & rankings[2])
overlap_12 = len(rankings[1] & rankings[2])
avg_overlap = (overlap_01 + overlap_02 + overlap_12) / 3
print(f"\nStability check - avg top-5 feature overlap across 3 bootstrap resamples: {avg_overlap:.2f} / 5")

with open("results/m6_shap_results.txt", "w") as fh:
    fh.write("M6 SHAP Explainability (LinearExplainer on sklearn LR)\n")
    fh.write(f"Feature set/split: loaded from {common.M3_FEATURES_PATH} and "
             f"{common.SPLIT_PATH} - the exact corrected M3 feature table and the "
             "same shared split used by M1/M3, NOT independently recomputed.\n")
    fh.write(f"sklearn LR test accuracy: {test_acc:.4f}\n")
    fh.write(f"Fidelity (max abs error, SHAP sum vs model margin): {fidelity_max_err:.8f}\n\n")
    fh.write("Top 10 features by mean |SHAP value|:\n")
    fh.write("\n".join(top_features_report))
    fh.write(f"\n\nStability - avg top-5 feature overlap across 3 bootstrap resamples: {avg_overlap:.2f} / 5\n")

print("\nSaved to results/m6_shap_results.txt")
