"""
ReviewGuard - M3 Network/Campaign Features (v2, corrected)
Fixes vs v1:
  - PageRank now uses BOTH edge directions (src->dst AND dst->src). Degree and
    connected components are direction-insensitive in GraphFrames so the
    original one-directional (row<col) edges were fine for those - but
    PageRank is directional, so v1's PageRank was an artifact of numeric
    review-ID ordering, not real structure.
  - Uses the SHARED split (data/split.csv), identical to M1's, instead of its
    own randomSplit call.
  - Reports fraud-class F1/precision/recall/AUPRC/confusion matrix.
  - Adds a deg_rur-ALONE diagnostic: trains LR on just that one feature to
    test whether it alone explains most of the AUC jump (a real possibility,
    since YelpChi's labels come from Yelp's own filter, which may itself lean
    on reviewer-activity signals).
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("ReviewGuard-M3-v2") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "32") \
    .config("spark.driver.memory", "3g") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")

from graphframes import GraphFrame

mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()
labels = mat["label"].flatten()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels.astype(int)
pdf["row_id"] = np.arange(n)

def edges_from_sparse_one_direction(mat_sparse):
    coo = mat_sparse.tocoo()
    mask = coo.row < coo.col
    return pd.DataFrame({"src": coo.row[mask], "dst": coo.col[mask]})

def edges_from_sparse_both_directions(mat_sparse):
    one_dir = edges_from_sparse_one_direction(mat_sparse)
    reversed_dir = pd.DataFrame({"src": one_dir["dst"], "dst": one_dir["src"]})
    return pd.concat([one_dir, reversed_dir], ignore_index=True)

edges_rur = edges_from_sparse_one_direction(mat["net_rur"])
edges_rsr = edges_from_sparse_one_direction(mat["net_rsr"])
edges_rtr = edges_from_sparse_one_direction(mat["net_rtr"])

print(f"Edges -> RUR: {len(edges_rur)}, RSR: {len(edges_rsr)}, RTR: {len(edges_rtr)}")

vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n)}))

def degree_feature(edges_pdf, name):
    e = spark.createDataFrame(edges_pdf)
    g = GraphFrame(vertices, e)
    return g.degrees.withColumnRenamed("degree", name)

deg_rur = degree_feature(edges_rur, "deg_rur")
deg_rsr = degree_feature(edges_rsr, "deg_rsr")
deg_rtr = degree_feature(edges_rtr, "deg_rtr")

# PageRank: BOTH directions now (fix for v1's artificial row<col direction bug)
# PageRank scoped to R-U-R + R-T-R only (excludes R-S-R, the densest
# relation at 3.4M edges, which M4 already showed collapses the graph
# into one giant component without adding real reviewer-behavior signal)
edges_rur_one_dir = edges_from_sparse_one_direction(mat["net_rur"])
edges_rtr_one_dir = edges_from_sparse_one_direction(mat["net_rtr"])
edges_rur_rtr_one_dir = pd.concat([edges_rur_one_dir, edges_rtr_one_dir], ignore_index=True)
edges_rur_rtr_reversed = pd.DataFrame({"src": edges_rur_rtr_one_dir["dst"], "dst": edges_rur_rtr_one_dir["src"]})
edges_homo_bidir = pd.concat([edges_rur_rtr_one_dir, edges_rur_rtr_reversed], ignore_index=True)
e_homo_bidir = spark.createDataFrame(edges_homo_bidir)
g_homo_bidir = GraphFrame(vertices, e_homo_bidir)
pr = g_homo_bidir.pageRank(resetProbability=0.15, maxIter=5)
pagerank_df = pr.vertices.select("id", "pagerank")

main_df = spark.createDataFrame(pdf).withColumnRenamed("row_id", "id")
merged = main_df.join(deg_rur, "id", "left") \
                 .join(deg_rsr, "id", "left") \
                 .join(deg_rtr, "id", "left") \
                 .join(pagerank_df, "id", "left") \
                 .fillna(0, subset=["deg_rur", "deg_rsr", "deg_rtr", "pagerank"])

graph_cols = cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]

split_pdf = pd.read_csv("data/split.csv").rename(columns={"row_id": "id"})
split_sdf = spark.createDataFrame(split_pdf)
merged = merged.join(split_sdf, "id", "inner")

train_df = merged.filter(merged.split == "train")
test_df = merged.filter(merged.split == "test")
print(f"Train: {train_df.count()}  Test: {test_df.count()}")

def fit_and_eval(feature_cols, train_df, test_df, label_str):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    train_v = assembler.transform(train_df)
    test_v = assembler.transform(test_df)
    scaler_model = scaler.fit(train_v)
    train_s = scaler_model.transform(train_v)
    test_s = scaler_model.transform(test_v)
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
    model = lr.fit(train_s)
    preds = model.transform(test_s)

    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)
    auprc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR").evaluate(preds)
    fraud_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0).evaluate(preds)
    fraud_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0).evaluate(preds)
    fraud_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0).evaluate(preds)
    weighted_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(preds)

    print(f"\n=== {label_str} ===")
    print(f"ROC-AUC: {auc:.4f}  AUPRC: {auprc:.4f}")
    print(f"Fraud F1: {fraud_f1:.4f}  Fraud precision: {fraud_precision:.4f}  Fraud recall: {fraud_recall:.4f}")
    print(f"Weighted F1 (reference only): {weighted_f1:.4f}")
    return dict(auc=auc, auprc=auprc, fraud_f1=fraud_f1, fraud_precision=fraud_precision, fraud_recall=fraud_recall, weighted_f1=weighted_f1)

# Full M3 feature set
m3_results = fit_and_eval(graph_cols, train_df, test_df, "M3: handcrafted + graph features (v2, PageRank fixed, shared split)")

# Diagnostic: deg_rur ALONE - tests whether this one feature explains most of the jump
degrur_results = fit_and_eval(["deg_rur"], train_df, test_df, "DIAGNOSTIC: deg_rur ALONE")

with open("results/m3_network_results.txt", "w") as fh:
    fh.write("M3 v2 (handcrafted + graph features: degree x3 + PageRank, PageRank direction fixed, shared split)\n")
    for k, v in m3_results.items():
        fh.write(f"{k}: {v:.4f}\n")
    fh.write("\nDIAGNOSTIC - deg_rur ALONE (tests if this one feature explains most of the M1->M3 jump):\n")
    for k, v in degrur_results.items():
        fh.write(f"{k}: {v:.4f}\n")

spark.stop()
