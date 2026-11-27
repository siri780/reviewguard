"""
ReviewGuard - M3b: deg_rur ablation

Five configurations on the same shared split:
  (a) M1 (32 handcrafted features)
  (b) M1 + deg_rur only (33 features)
  (c) Full M3 (32 handcrafted + deg_rur + deg_rsr + deg_rtr + pagerank, 36 features)
  (d) M3 minus deg_rur (35 features)
  (e) deg_rur alone (1 feature; diagnostic only, not used to judge its role inside M3)

Run from the repository root AFTER m3_network_features.py:  python src/m3b_ablation.py
(needs results/_m3_features.csv and results/_shared_split_row_ids.csv)
"""
import pandas as pd
from pyspark.sql import SparkSession

import common

merged_pdf = pd.read_csv(common.M3_FEATURES_PATH)
train_pdf, test_pdf = common.stratified_split(merged_pdf, reuse=True)

handcrafted_cols = [f"f{i}" for i in range(32)]
graph_cols = ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]

configs = {
    "(a) M1 only (32 handcrafted)": handcrafted_cols,
    "(b) M1 + deg_rur only (33 features)": handcrafted_cols + ["deg_rur"],
    "(c) Full M3 (36 features)": handcrafted_cols + graph_cols,
    "(d) M3 minus deg_rur (35 features)": handcrafted_cols + [c for c in graph_cols if c != "deg_rur"],
    "(e) deg_rur alone (1 feature)": ["deg_rur"],
}

spark = SparkSession.builder.appName("ReviewGuard-M3b-Ablation").master("local[*]").getOrCreate()

results = {}
for name, feat_cols in configs.items():
    metrics = common.train_eval_lr(spark, feat_cols, train_pdf, test_pdf)
    results[name] = metrics
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

with open("results/m3b_ablation_results.txt", "w") as fh:
    fh.write("M3b: deg_rur ablation (same shared split/test set for all five configs)\n")
    fh.write("=" * 70 + "\n")
    for name, metrics in results.items():
        fh.write("\n" + common.format_results(name, metrics))
        fh.write("-" * 70 + "\n")

    a = results["(a) M1 only (32 handcrafted)"]
    b = results["(b) M1 + deg_rur only (33 features)"]
    c = results["(c) Full M3 (36 features)"]
    d = results["(d) M3 minus deg_rur (35 features)"]
    fh.write(
        "\nDerived (change = after - before):\n"
        f"Removing deg_rur from the full model lowers ROC-AUC from {c['roc_auc']:.4f} to {d['roc_auc']:.4f} "
        f"({d['roc_auc'] - c['roc_auc']:+.4f}) and lowers fraud-class F1 from {c['fraud_f1']:.4f} to "
        f"{d['fraud_f1']:.4f} ({d['fraud_f1'] - c['fraud_f1']:+.4f}).\n"
        f"Adding deg_rur alone to M1 raises ROC-AUC from {a['roc_auc']:.4f} to {b['roc_auc']:.4f} "
        f"({b['roc_auc'] - a['roc_auc']:+.4f}).\n"
        "These two comparisons, not the standalone deg_rur diagnostic (e), state deg_rur's "
        "contribution within the model.\n"
    )

print("\nWrote results/m3b_ablation_results.txt")
spark.stop()
