"""
ReviewGuard - M1 Baseline (v2, corrected)
- Uses the SHARED split (data/split.csv) instead of its own randomSplit,
  so M1/M3/M6 are guaranteed to share identical train/test membership.
- Reports fraud-class F1 + AUPRC + confusion matrix, not just weighted F1
  (weighted F1 is misleading on a 14.5% minority-class problem).
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()
labels = mat["label"].flatten()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels.astype(int)
pdf["row_id"] = np.arange(n)

split_pdf = pd.read_csv("data/split.csv")
pdf = pdf.merge(split_pdf, on="row_id", how="inner")
assert len(pdf) == n, f"Split join dropped rows: {len(pdf)} != {n}"

spark = SparkSession.builder.appName("ReviewGuard-M1-v2").master("local[*]").getOrCreate()
sdf = spark.createDataFrame(pdf)

train_df = sdf.filter(sdf.split == "train")
test_df = sdf.filter(sdf.split == "test")
print(f"Train: {train_df.count()}  Test: {test_df.count()}")

assembler = VectorAssembler(inputCols=cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

train_v = assembler.transform(train_df)
test_v = assembler.transform(test_df)
scaler_model = scaler.fit(train_v)
train_s = scaler_model.transform(train_v)
test_s = scaler_model.transform(test_v)

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
model = lr.fit(train_s)
preds = model.transform(test_s)

auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auprc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
fraud_f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0)
fraud_precision_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
fraud_recall_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
weighted_f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

auc = auc_eval.evaluate(preds)
auprc = auprc_eval.evaluate(preds)
fraud_f1 = fraud_f1_eval.evaluate(preds)
fraud_precision = fraud_precision_eval.evaluate(preds)
fraud_recall = fraud_recall_eval.evaluate(preds)
weighted_f1 = weighted_f1_eval.evaluate(preds)

cm = preds.groupBy("label", "prediction").count().collect()
cm_dict = {(int(r["label"]), int(r["prediction"])): r["count"] for r in cm}
tp = cm_dict.get((1, 1), 0)
fp = cm_dict.get((0, 1), 0)
fn = cm_dict.get((1, 0), 0)
tn = cm_dict.get((0, 0), 0)

print(f"\n=== M1 Baseline Results (v2, corrected) ===")
print(f"ROC-AUC: {auc:.4f}")
print(f"AUPRC: {auprc:.4f}")
print(f"Fraud-class F1: {fraud_f1:.4f}")
print(f"Fraud precision: {fraud_precision:.4f}")
print(f"Fraud recall: {fraud_recall:.4f}")
print(f"Weighted F1 (previous headline metric - inflated by majority class): {weighted_f1:.4f}")
print(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")

with open("results/m1_baseline_results.txt", "w") as fh:
    fh.write("M1 Baseline v2 (Logistic Regression, 32 handcrafted features, SHARED split)\n")
    fh.write(f"Dataset: YelpChi, {n} reviews, fraud rate {labels.mean():.4f}\n")
    fh.write(f"ROC-AUC: {auc:.4f}\n")
    fh.write(f"AUPRC: {auprc:.4f}\n")
    fh.write(f"Fraud-class F1: {fraud_f1:.4f}\n")
    fh.write(f"Fraud precision: {fraud_precision:.4f}\n")
    fh.write(f"Fraud recall: {fraud_recall:.4f}\n")
    fh.write(f"Weighted F1 (for reference only - inflated by majority class): {weighted_f1:.4f}\n")
    fh.write(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}\n")

spark.stop()
