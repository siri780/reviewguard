"""
ReviewGuard - M7 Scalability Benchmark
Two honest scalability dimensions given YelpChi's ~46K-review size cap:
  (A) Runtime vs. data volume (fraction of dataset, fixed cores)
  (B) Runtime vs. available compute (local[N] cores, fixed full data)
"""
import time
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()
labels = mat["label"].flatten()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

full_pdf = pd.DataFrame(features, columns=cols)
full_pdf["label"] = labels.astype(int)

def run_pipeline(spark, pdf):
    t0 = time.time()
    sdf = spark.createDataFrame(pdf)
    train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)
    assembler = VectorAssembler(inputCols=cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    train_v = assembler.transform(train_df)
    test_v = assembler.transform(test_df)
    scaler_model = scaler.fit(train_v)
    train_s = scaler_model.transform(train_v)
    test_s = scaler_model.transform(test_v)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    model = lr.fit(train_s)
    _ = model.transform(test_s).count()
    return time.time() - t0

results = []

spark_a = SparkSession.builder.appName("Scalability-DataVolume").master("local[*]").getOrCreate()
spark_a.sparkContext.setLogLevel("ERROR")
print(f"Detected cores (local[*]): {spark_a.sparkContext.defaultParallelism}")

for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
    sample_pdf = full_pdf.sample(frac=frac, random_state=42).reset_index(drop=True)
    elapsed = run_pipeline(spark_a, sample_pdf)
    rows = len(sample_pdf)
    print(f"Data fraction {frac:.1f} ({rows} rows): {elapsed:.2f}s")
    results.append(("data_volume", frac, rows, None, elapsed))

spark_a.stop()

max_cores = 4
for n_cores in [1, 2, max_cores]:
    spark_b = SparkSession.builder.appName(f"Scalability-Cores-{n_cores}").master(f"local[{n_cores}]").getOrCreate()
    spark_b.sparkContext.setLogLevel("ERROR")
    elapsed = run_pipeline(spark_b, full_pdf)
    print(f"Cores={n_cores}: {elapsed:.2f}s")
    results.append(("cores", n_cores, len(full_pdf), n_cores, elapsed))
    spark_b.stop()

with open("results/m7_scalability_results.txt", "w") as fh:
    fh.write("M7 Scalability Benchmark\n\n")
    fh.write("(A) Runtime vs data volume (local[*] cores):\n")
    for kind, x, rows, cores, elapsed in results:
        if kind == "data_volume":
            fh.write(f"  fraction={x:.1f}, rows={rows}: {elapsed:.2f}s\n")
    fh.write("\n(B) Runtime vs available cores (full data):\n")
    for kind, x, rows, cores, elapsed in results:
        if kind == "cores":
            fh.write(f"  cores={cores}: {elapsed:.2f}s\n")
    fh.write("\nNote: YelpChi caps at ~46K reviews; full 1M+ scale benchmarking\n")
    fh.write("deferred to the raw-text dataset pending author's reply.\n")

print("\nSaved to results/m7_scalability_results.txt")
