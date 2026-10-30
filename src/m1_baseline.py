"""
ReviewGuard - M1 Baseline (corrected)
Logistic Regression on the 32 handcrafted YelpChi features.

Fixes applied after review:
  - Uses a stratified split persisted to results/_shared_split_row_ids.csv
    (see src/common.py) so M3 and the ablation script in m3b_ablation.py
    evaluate on the IDENTICAL held-out rows as this script, not an
    independently-drawn one.
  - Reports AUPRC, fraud-class precision/recall/F1, and a confusion matrix
    alongside ROC-AUC and weighted F1 (weighted F1 alone is dominated by the
    majority/genuine class at this 14.5% fraud rate).
"""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

import common

mat, features, labels = common.load_mat()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels
pdf["row_id"] = np.arange(n)

print(f"Loaded {n} reviews, {features.shape[1]} features")
print(f"Fraud rate: {labels.mean():.4f}")

train_pdf, test_pdf = common.stratified_split(pdf, reuse=True)
print(f"Train: {len(train_pdf)}  Test: {len(test_pdf)}  "
      f"(shared split saved to {common.SPLIT_PATH})")

spark = SparkSession.builder.appName("ReviewGuard-M1").master("local[*]").getOrCreate()

metrics = common.train_eval_lr(spark, cols, train_pdf, test_pdf)

print("\n=== M1 Baseline Results (corrected) ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

header = (
    "M1 Baseline (Logistic Regression, 32 handcrafted features)\n"
    f"Dataset: YelpChi, {n} reviews, fraud rate {labels.mean():.4f}\n"
    f"Shared stratified split (row ids in {common.SPLIT_PATH}); "
    "M3 and the ablation script reuse this exact split."
)
with open("results/m1_baseline_results.txt", "w") as fh:
    fh.write(common.format_results(header, metrics))

spark.stop()
