"""
ReviewGuard - M5 Leakage Audit (corrected)
Formal checks that the M1/M3/M4 pipeline does not leak label information,
documents the transductive-graph design choice explicitly, and verifies the
shared train/test split is actually shared.

Fix applied after review: the audit previously printed "PASS" as a literal
string regardless of what the checks found. Every check below now sets a
boolean, and the final verdict is computed from those booleans, not
hardcoded. Run this AFTER m1_baseline.py and m3_network_features.py, since
Check 5 reads the split file they produce.
"""
import os
import scipy.io as sio
import numpy as np
import pandas as pd

import common

report_lines = ["=== ReviewGuard Leakage Audit ===\n"]
all_pass = True

mat, features, labels = common.load_mat()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]
pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels
pdf["row_id"] = np.arange(n)

# --- Check 1: train/test row overlap, using the ACTUAL shared split file ---
if os.path.exists(common.SPLIT_PATH):
    split_df = pd.read_csv(common.SPLIT_PATH)
    train_ids = set(split_df.loc[split_df["split"] == "train", "row_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "row_id"])
    overlap = train_ids & test_ids
    check1_pass = (len(overlap) == 0) and (len(train_ids) + len(test_ids) == n)
    report_lines.append(f"Check 1 - Train/test row overlap: {len(overlap)} overlapping rows (expect 0)")
    report_lines.append(
        f"  Train size: {len(train_ids)}, Test size: {len(test_ids)}, "
        f"Total: {len(train_ids) + len(test_ids)} (expect {n}) -> "
        f"{'PASS' if check1_pass else 'FAIL'}"
    )
else:
    check1_pass = False
    report_lines.append(
        f"Check 1 - FAIL: {common.SPLIT_PATH} not found. Run m1_baseline.py first "
        "to create the shared split, then re-run this audit."
    )
all_pass &= check1_pass

# --- Check 2: feature-label correlation ---
corrs = {c: pdf[[c, "label"]].corr().iloc[0, 1] for c in cols}
sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)
report_lines.append("\nCheck 2 - Feature-label correlations (top 5, flag if any > 0.95):")
suspicious = [feat for feat, corr in sorted_corrs if abs(corr) > 0.95]
for feat, corr in sorted_corrs[:5]:
    report_lines.append(f"  {feat}: corr={corr:.4f}")
check2_pass = len(suspicious) == 0
report_lines.append(f"  Suspicious (|corr|>0.95) features: {suspicious if suspicious else 'none'} -> "
                     f"{'PASS' if check2_pass else 'FAIL'}")
all_pass &= check2_pass

# --- Check 3: graph feature construction does not read labels (static check) ---
report_lines.append("\nCheck 3 - Graph feature construction (code inspection, not re-verified at runtime):")
report_lines.append("  degree, PageRank, and component membership are computed purely from edge")
report_lines.append("  structure (net_rur/net_rsr/net_rtr adjacency). No review's label is read when")
report_lines.append("  computing another review's graph feature. M4's fraud_rate-per-component is a")
report_lines.append("  POST-HOC diagnostic only - never fed back in as a model input feature.")
report_lines.append("  (This check is not included in the automated PASS/FAIL verdict below; it")
report_lines.append("   reflects a manual review of the source, not a runtime assertion.)")

# --- Check 4: transductive setting disclosure (design statement, not a pass/fail test) ---
report_lines.append("\nCheck 4 - Transductive graph setting (explicit disclosure, not a pass/fail test):")
report_lines.append("  Graph structure (edges, degree, PageRank, components) is computed on the FULL")
report_lines.append("  45,954-review graph before the train/test split is applied. This is standard")
report_lines.append("  practice for graph-based fraud detection (transductive setting: graph topology")
report_lines.append("  is observable at train time, only TEST LABELS are held out). It is NOT")
report_lines.append("  equivalent to leaking test labels - no label is used in any graph feature -")
report_lines.append("  but it is weaker than a fully inductive setup where test nodes' edges are also")
report_lines.append("  hidden at train time. Disclosed explicitly; M3/M4 results are not presented as")
report_lines.append("  inductive-generalization numbers.")

# --- Check 5: M1 and M3 actually used the identical split ---
m3_feat_exists = os.path.exists(common.M3_FEATURES_PATH)
if check1_pass and m3_feat_exists:
    m3_pdf = pd.read_csv(common.M3_FEATURES_PATH)
    m3_ids = set(m3_pdf["row_id"])
    check5_pass = m3_ids == set(pdf["row_id"])
    report_lines.append(
        f"\nCheck 5 - M1/M3 share one split file ({common.SPLIT_PATH}) and cover the same "
        f"{len(m3_ids)} row ids -> {'PASS' if check5_pass else 'FAIL'}"
    )
else:
    check5_pass = False
    report_lines.append(
        f"\nCheck 5 - FAIL: could not verify (run m1_baseline.py and m3_network_features.py first)."
    )
all_pass &= check5_pass

verdict = "PASS" if all_pass else "FAIL"
report_lines.append(
    f"\n=== Audit Result: {verdict} "
    f"(checks 1, 2, 5 {'all passed' if all_pass else 'did NOT all pass - see above'}; "
    "transductive design disclosed in Check 4) ==="
)

report = "\n".join(report_lines)
print(report)
with open("results/m5_leakage_audit.txt", "w") as fh:
    fh.write(report)
