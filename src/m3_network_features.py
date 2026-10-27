"""
ReviewGuard - M3 Network/Campaign Features
Adds graph-structural features (degree per relation + PageRank) from the
R-U-R, R-S-R, R-T-R review graphs on top of the M1 handcrafted features.
No label information is used in these features (leakage-safe).
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("ReviewGuard-M3") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")

from graphframes import GraphFrame

# ---- 1. Load data ----
mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()
labels = mat["label"].flatten()
n = features.shape[0]

cols = [f"f{i}" for i in range(features.shape[1])]
pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels.astype(int)
pdf["row_id"] = np.arange(n)

# ---- 2. Build edge lists from each relation graph (undirected, i<j only) ----
def edges_from_sparse(mat_sparse):
    coo = mat_sparse.tocoo()
    mask = coo.row < coo.col
    return pd.DataFrame({"src": coo.row[mask], "dst": coo.col[mask]})

edges_rur = edges_from_sparse(mat["net_rur"])
edges_rsr = edges_from_sparse(mat["net_rsr"])
edges_rtr = edges_from_sparse(mat["net_rtr"])

print(f"Edges -> RUR: {len(edges_rur)}, RSR: {len(edges_rsr)}, RTR: {len(edges_rtr)}")

vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n)}))

def degree_feature(edges_pdf, name):
    e = spark.createDataFrame(edges_pdf)
    g = GraphFrame(vertices, e)
    deg = g.degrees.withColumnRenamed("degree", name)
    return deg

deg_rur = degree_feature(edges_rur, "deg_rur")
deg_rsr = degree_feature(edges_rsr, "deg_rsr")
deg_rtr = degree_feature(edges_rtr, "deg_rtr")

# ---- 3. PageRank on the combined (homo) graph - a real distributed graph algorithm ----
edges_homo = edges_from_sparse(mat["homo"])
e_homo = spark.createDataFrame(edges_homo)
g_homo = GraphFrame(vertices, e_homo)
pr = g_homo.pageRank(resetProbability=0.15, maxIter=5)
pagerank_df = pr.vertices.select("id", "pagerank")

# ---- 4. Merge graph features with original features ----
main_df = spark.createDataFrame(pdf).withColumnRenamed("row_id", "id")
merged = main_df.join(deg_rur, "id", "left") \
                 .join(deg_rsr, "id", "left") \
                 .join(deg_rtr, "id", "left") \
                 .join(pagerank_df, "id", "left") \
                 .fillna(0, subset=["deg_rur", "deg_rsr", "deg_rtr", "pagerank"])

graph_cols = cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]

# ---- 5. Same leakage-safe split as M1 (seed=42) for a fair ablation comparison ----
train_df, test_df = merged.randomSplit([0.8, 0.2], seed=42)
print(f"Train: {train_df.count()}  Test: {test_df.count()}")

assembler = VectorAssembler(inputCols=graph_cols, outputCol="features_raw")
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
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

auc = auc_eval.evaluate(preds)
f1 = f1_eval.evaluate(preds)

print(f"\n=== M1 + Network (M3) Results ===")
print(f"AUC: {auc:.4f}")
print(f"F1:  {f1:.4f}")

with open("results/m3_network_results.txt", "w") as fh:
    fh.write("M3 (handcrafted features + graph-structural features: degree x3 + PageRank)\n")
    fh.write(f"AUC: {auc:.4f}\nF1: {f1:.4f}\n")

spark.stop()
