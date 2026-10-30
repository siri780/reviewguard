"""
ReviewGuard - M3b: deg_rur ablation (addresses review point #2)

The earlier "deg_rur alone" diagnostic only showed that one feature is weak
by itself - it does not show how much deg_rur contributes WITHIN the full
model. This script runs the four comparisons needed to actually answer that:

  (a) M1 (32 handcrafted features)                     -> baseline
  (b) M1 + deg_rur only (33 features)                   -> does this one
      feature alone explain most of the M1->M3 jump?
  (c) Full M3 (32 handcrafted + deg_rur + deg_rsr +
      deg_rtr + pagerank, 36 features)                  -> everything
  (d) M3 minus deg_rur (35 features: handcrafted +
      deg_rsr + deg_rtr + pagerank)                      -> does removing
      just deg_rur from the full model hurt much?
  (e) deg_rur alone (1 feature)                         -> kept from the
      earlier diagnostic for continuity, but NOT used on its own to draw
      conclusions about its role inside M3.

Run this AFTER m3_network_features.py (needs results/_m3_features.csv and
results/_shared_split_row_ids.csv, both written by that script).
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

    c_minus_d_auc = results["(c) Full M3 (36 features)"]["roc_auc"] - results["(d) M3 minus deg_rur (35 features)"]["roc_auc"]
    c_minus_d_f1 = results["(c) Full M3 (36 features)"]["fraud_f1"] - results["(d) M3 minus deg_rur (35 features)"]["fraud_f1"]
    a_minus_b_auc = results["(b) M1 + deg_rur only (33 features)"]["roc_auc"] - results["(a) M1 only (32 handcrafted)"]["roc_auc"]
    fh.write(
        f"\nDerived: removing deg_rur from full M3 changes ROC-AUC by {c_minus_d_auc:+.4f} "
        f"and fraud F1 by {c_minus_d_f1:+.4f}.\n"
        f"Derived: adding deg_rur alone to M1 changes ROC-AUC by {a_minus_b_auc:+.4f}.\n"
        "These two numbers - not the standalone deg_rur diagnostic - are what should be "
        "used to state deg_rur's actual contribution within the model.\n"
    )

print("\nWrote results/m3b_ablation_results.txt")
spark.stop()
