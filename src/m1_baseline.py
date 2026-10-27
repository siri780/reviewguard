"""
ReviewGuard - M1 Baseline
Logistic Regression on the 32 handcrafted YelpChi features.
Real labeled fraud data (Rayana & Akoglu, KDD'15 / DGL FraudYelpDataset).
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ---- 1. Load raw .mat data ----
mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()          # (45954, 32)
labels = mat["label"].flatten()                # (45954,)  0 = genuine, 1 = fraud

print(f"Loaded {features.shape[0]} reviews, {features.shape[1]} features")
print(f"Fraud rate: {labels.mean():.4f}")

# ---- 2. Build a pandas DF, then a Spark DF ----
cols = [f"f{i}" for i in range(features.shape[1])]
pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels.astype(int)
pdf["row_id"] = np.arange(len(pdf))

spark = SparkSession.builder.appName("ReviewGuard-M1").master("local[*]").getOrCreate()
sdf = spark.createDataFrame(pdf)

# ---- 3. Leakage-safe stratified split (80/20, fixed seed) ----
train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()}  Test: {test_df.count()}")

# ---- 4. Assemble + scale features ----
assembler = VectorAssembler(inputCols=cols, outputCol="features_raw")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

train_v = assembler.transform(train_df)
test_v = assembler.transform(test_df)
scaler_model = scaler.fit(train_v)
train_s = scaler_model.transform(train_v)
test_s = scaler_model.transform(test_v)

# ---- 5. Train Logistic Regression ----
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
model = lr.fit(train_s)
preds = model.transform(test_s)

# ---- 6. Evaluate ----
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

auc = auc_eval.evaluate(preds)
f1 = f1_eval.evaluate(preds)

print(f"\n=== M1 Baseline Results ===")
print(f"AUC: {auc:.4f}")
print(f"F1:  {f1:.4f}")

with open("results/m1_baseline_results.txt", "w") as fh:
    fh.write(f"M1 Baseline (Logistic Regression, 32 handcrafted features)\n")
    fh.write(f"Dataset: YelpChi, {features.shape[0]} reviews, fraud rate {labels.mean():.4f}\n")
    fh.write(f"AUC: {auc:.4f}\nF1: {f1:.4f}\n")

spark.stop()
