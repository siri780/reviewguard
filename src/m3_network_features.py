"""
ReviewGuard - M3 Network Features (corrected)

Fixes applied after review (each one previously present and now removed):
  1. PageRank now runs on BIDIRECTIONAL edges, not `row < col` one-way edges.
     PageRank is a directed algorithm; a one-way edge list starves most
     nodes of inbound rank mass regardless of real connectivity.
  2. PageRank is scoped to R-U-R + R-T-R ONLY. It previously ran on
     mat["homo"], the precomputed combined graph that also includes R-S-R
     (same product+rating/month) - a relation dense enough to put ~45,900 of
     45,954 reviews in one connected component (see m4_campaign_results.txt),
     which makes centrality on that graph closer to a popularity count than
     a campaign/network signal.
  3. Uses the SAME stratified split as M1 (src/common.py,
     results/_shared_split_row_ids.csv) instead of an independently drawn
     `randomSplit(seed=42)` - same seed does not guarantee the same rows
     once DataFrame construction/partitioning differs between scripts.
  4. Reports AUPRC + fraud-class precision/recall/F1 + confusion matrix, not
     just ROC-AUC and weighted F1.
  5. Saves the merged feature table (results/_m3_features.csv) so
     m3b_ablation.py and m6_shap_explainability.py reuse the IDENTICAL
     graph-feature values this script computed, rather than recomputing
     them (and risking silent drift) a second time.

No label information is used in any graph feature - degree and PageRank are
pure functions of edge structure (see m5_leakage_audit.py, Check 3).
"""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

import common

spark = SparkSession.builder.appName("ReviewGuard-M3") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")
from graphframes import GraphFrame  # noqa: E402  (after SparkSession config)

# ---- 1. Load data ----
mat, features, labels = common.load_mat()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

pdf = pd.DataFrame(features, columns=cols)
pdf["label"] = labels
pdf["row_id"] = np.arange(n)

vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n)}))


def degree_feature(relation_key, name):
    """Plain undirected degree: one row per unordered pair is enough."""
    edges = common.edges_from_sparse(mat[relation_key], bidirectional=False)
    e = spark.createDataFrame(edges)
    g = GraphFrame(vertices, e)
    return g.degrees.withColumnRenamed("degree", name)


# ---- 2. Degree per relation (unchanged: R-U-R, R-S-R, R-T-R all included -
#          degree alone doesn't have R-S-R's density problem, PageRank does) ----
deg_rur = degree_feature("net_rur", "deg_rur")
deg_rsr = degree_feature("net_rsr", "deg_rsr")
deg_rtr = degree_feature("net_rtr", "deg_rtr")

# ---- 3. PageRank: bidirectional edges, R-U-R + R-T-R only (R-S-R excluded) ----
edges_rur_bi = common.edges_from_sparse(mat["net_rur"], bidirectional=True)
edges_rtr_bi = common.edges_from_sparse(mat["net_rtr"], bidirectional=True)
edges_pr = pd.concat([edges_rur_bi, edges_rtr_bi], ignore_index=True).drop_duplicates()
print(f"PageRank edges (bidirectional, R-U-R+R-T-R only): {len(edges_pr)}")

e_pr = spark.createDataFrame(edges_pr)
g_pr = GraphFrame(vertices, e_pr)
pr = g_pr.pageRank(resetProbability=0.15, maxIter=10)
pagerank_df = pr.vertices.select("id", "pagerank")

# ---- 4. Merge graph features with the 32 handcrafted features ----
main_df = spark.createDataFrame(pdf).withColumnRenamed("row_id", "id")
merged = main_df.join(deg_rur, "id", "left") \
                 .join(deg_rsr, "id", "left") \
                 .join(deg_rtr, "id", "left") \
                 .join(pagerank_df, "id", "left") \
                 .fillna(0, subset=["deg_rur", "deg_rsr", "deg_rtr", "pagerank"])

graph_cols = cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]

merged_pdf = merged.toPandas().rename(columns={"id": "row_id"})
merged_pdf.to_csv(common.M3_FEATURES_PATH, index=False)
print(f"Saved merged feature table to {common.M3_FEATURES_PATH} "
      "(reused by m3b_ablation.py and m6_shap_explainability.py)")

# ---- 5. Same shared stratified split as M1 ----
train_pdf, test_pdf = common.stratified_split(merged_pdf, reuse=True)
print(f"Train: {len(train_pdf)}  Test: {len(test_pdf)}  (shared split with M1)")

metrics = common.train_eval_lr(spark, graph_cols, train_pdf, test_pdf)

print("\n=== M3 (M1 + network features) Results (corrected) ===")
for k, v in metrics.items():
    print(f"{k}: {v}")

header = (
    "M3 (handcrafted features + graph-structural features: degree x3 + PageRank)\n"
    "PageRank: bidirectional edges, scoped to R-U-R+R-T-R (R-S-R excluded).\n"
    "Same shared stratified split as M1 (src/common.py)."
)
with open("results/m3_network_results.txt", "w") as fh:
    fh.write(common.format_results(header, metrics))

spark.stop()
