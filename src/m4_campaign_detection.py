"""
ReviewGuard - M4 Campaign Detection (v2)
Uses ONLY the R-U-R (same reviewer) relation for connected components,
since the combined graph collapses into one giant component (R-S-R/R-T-R
are too dense - same-product-rating/month connects almost everyone).
R-U-R alone isolates genuine repeat-reviewer clusters: a real campaign signal.
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

spark = SparkSession.builder.appName("ReviewGuard-M4-Campaigns-v2") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")

mat = sio.loadmat("data/YelpChi.mat")
labels = mat["label"].flatten()
n = len(labels)

def edges_from_sparse(mat_sparse):
    coo = mat_sparse.tocoo()
    mask = coo.row < coo.col
    return pd.DataFrame({"src": coo.row[mask], "dst": coo.col[mask]})

edges_rur = edges_from_sparse(mat["net_rur"])
vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n), "label": labels.astype(int)}))
e_rur = spark.createDataFrame(edges_rur)

g = GraphFrame(vertices, e_rur)
cc = g.connectedComponents()

comp_stats = cc.groupBy("component").agg(
    F.count("*").alias("comp_size"),
    F.avg("label").alias("fraud_rate")
).orderBy(F.desc("comp_size"))

print("=== Top 15 largest R-U-R connected components (reviewer clusters) ===")
comp_stats.show(15, truncate=False)

total_components = comp_stats.count()
singleton_components = comp_stats.filter(F.col("comp_size") == 1).count()
multi_review_components = comp_stats.filter(F.col("comp_size") >= 2).count()

campaign_threshold_size = 2
campaign_threshold_fraud = 0.5
campaigns = comp_stats.filter(
    (F.col("comp_size") >= campaign_threshold_size) & (F.col("fraud_rate") >= campaign_threshold_fraud)
)
n_campaigns = campaigns.count()
reviews_in_campaigns = campaigns.agg(F.sum("comp_size")).collect()[0][0] or 0

overall_fraud_rate = labels.mean()
multi_review_fraud_rate = comp_stats.filter(F.col("comp_size") >= 2).agg(F.avg("fraud_rate")).collect()[0][0]

print(f"\nTotal reviewer clusters (R-U-R components): {total_components}")
print(f"Single-review reviewers (no repeat activity): {singleton_components}")
print(f"Multi-review reviewer clusters (size>=2): {multi_review_components}")
print(f"Avg fraud rate within multi-review clusters: {multi_review_fraud_rate:.4f}")
print(f"Overall dataset fraud rate: {overall_fraud_rate:.4f}")
print(f"Flagged campaigns (size>={campaign_threshold_size}, fraud_rate>={campaign_threshold_fraud}): {n_campaigns}")
print(f"Reviews covered by flagged campaigns: {reviews_in_campaigns}")

with open("results/m4_campaign_results.txt", "w") as fh:
    fh.write("M4 Campaign Detection v2 (connected components on R-U-R relation only)\n")
    fh.write("Note: combined R-U-R+R-S-R+R-T-R graph collapses into one giant component\n")
    fh.write("(45900/45954 nodes) because R-S-R/R-T-R are too dense; R-U-R alone isolates\n")
    fh.write("genuine repeat-reviewer clusters, a real campaign signal.\n\n")
    fh.write(f"Total reviewer clusters: {total_components}\n")
    fh.write(f"Single-review reviewers: {singleton_components}\n")
    fh.write(f"Multi-review clusters (size>=2): {multi_review_components}\n")
    fh.write(f"Avg fraud rate within multi-review clusters: {multi_review_fraud_rate:.4f}\n")
    fh.write(f"Overall dataset fraud rate: {overall_fraud_rate:.4f}\n")
    fh.write(f"Flagged campaigns: {n_campaigns}\n")
    fh.write(f"Reviews covered by flagged campaigns: {reviews_in_campaigns}\n")

spark.stop()