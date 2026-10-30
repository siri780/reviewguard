"""
ReviewGuard - M4 Campaign Detection (corrected)
Uses ONLY the R-U-R (same reviewer) relation for connected components, since
the combined graph collapses into one giant component (R-S-R/R-T-R are too
dense - same-product-rating/month connects almost everyone).

Fix applied after review: the earlier version only reported the UNWEIGHTED
mean of each component's fraud rate across multi-review components. That
number treats a 2-review component and a 50-review component identically,
so it cannot by itself show whether fraud concentrates in small or large
clusters - it needs to be compared against the REVIEW-WEIGHTED fraud rate
(i.e. total fraudulent reviews / total reviews, within multi-review
clusters), which this script now also computes.

This script makes no predictive claim and is not evaluated on a held-out
set: "campaigns" are components whose fraud_rate (computed from the known
label) crosses a threshold, so fraud_rate-per-component is a post-hoc,
label-dependent diagnostic, never a model input feature.
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

spark = SparkSession.builder.appName("ReviewGuard-M4-Campaigns") \
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
    F.sum("label").alias("fraud_count"),
    F.avg("label").alias("fraud_rate"),
).orderBy(F.desc("comp_size"))

print("=== Top 15 largest R-U-R connected components (reviewer clusters) ===")
comp_stats.show(15, truncate=False)

total_components = comp_stats.count()
singleton_components = comp_stats.filter(F.col("comp_size") == 1).count()
multi = comp_stats.filter(F.col("comp_size") >= 2)
multi_review_components = multi.count()

overall_fraud_rate = float(labels.mean())

# Unweighted: mean of each component's own fraud rate (old metric).
multi_unweighted_fraud_rate = multi.agg(F.avg("fraud_rate")).collect()[0][0] or 0.0

# Review-weighted: total fraud reviews / total reviews, across multi-review
# clusters - this is the number that actually answers "do big clusters skew
# genuine or fraudulent, weighted by how many reviews they contain".
multi_totals = multi.agg(
    F.sum("fraud_count").alias("fraud_sum"), F.sum("comp_size").alias("size_sum")
).collect()[0]
multi_weighted_fraud_rate = (multi_totals["fraud_sum"] / multi_totals["size_sum"]) if multi_totals["size_sum"] else 0.0

# Same two numbers restricted to the 15 largest components specifically,
# since "the largest clusters are genuine" is a claim about the largest
# clusters, not about all multi-review clusters.
top15 = comp_stats.orderBy(F.desc("comp_size")).limit(15)
top15_totals = top15.agg(
    F.sum("fraud_count").alias("fraud_sum"), F.sum("comp_size").alias("size_sum")
).collect()[0]
top15_weighted_fraud_rate = (top15_totals["fraud_sum"] / top15_totals["size_sum"]) if top15_totals["size_sum"] else 0.0

campaign_threshold_size = 2
campaign_threshold_fraud = 0.5
campaigns = comp_stats.filter(
    (F.col("comp_size") >= campaign_threshold_size) & (F.col("fraud_rate") >= campaign_threshold_fraud)
)
n_campaigns = campaigns.count()
reviews_in_campaigns = campaigns.agg(F.sum("comp_size")).collect()[0][0] or 0

print(f"\nTotal reviewer clusters (R-U-R components): {total_components}")
print(f"Single-review reviewers (no repeat activity): {singleton_components}")
print(f"Multi-review reviewer clusters (size>=2): {multi_review_components}")
print(f"Overall dataset fraud rate: {overall_fraud_rate:.4f}")
print(f"Multi-review clusters - UNWEIGHTED mean component fraud rate: {multi_unweighted_fraud_rate:.4f}")
print(f"Multi-review clusters - REVIEW-WEIGHTED fraud rate: {multi_weighted_fraud_rate:.4f}")
print(f"15 largest clusters - REVIEW-WEIGHTED fraud rate: {top15_weighted_fraud_rate:.4f}")
print(f"Flagged campaigns (size>={campaign_threshold_size}, fraud_rate>={campaign_threshold_fraud}): {n_campaigns}")
print(f"Reviews covered by flagged campaigns: {reviews_in_campaigns}")

with open("results/m4_campaign_results.txt", "w") as fh:
    fh.write("M4 Campaign Detection (connected components on R-U-R relation only)\n")
    fh.write("Note: combined R-U-R+R-S-R+R-T-R graph collapses into one giant component\n")
    fh.write("(45900/45954 nodes) because R-S-R/R-T-R are too dense; R-U-R alone isolates\n")
    fh.write("genuine repeat-reviewer clusters.\n\n")
    fh.write("Retrospective, label-dependent diagnostic, not a detector: 'campaigns' are\n")
    fh.write("defined using the known fraud_rate per component, so there is no held-out\n")
    fh.write("evaluation here and no claim about unseen reviews.\n\n")
    fh.write(f"Total reviewer clusters: {total_components}\n")
    fh.write(f"Single-review reviewers: {singleton_components}\n")
    fh.write(f"Multi-review clusters (size>=2): {multi_review_components}\n")
    fh.write(f"Overall dataset fraud rate: {overall_fraud_rate:.4f}\n")
    fh.write(f"Multi-review clusters - unweighted mean component fraud rate: {multi_unweighted_fraud_rate:.4f}\n")
    fh.write(f"Multi-review clusters - review-weighted fraud rate: {multi_weighted_fraud_rate:.4f}\n")
    fh.write(f"15 largest clusters - review-weighted fraud rate: {top15_weighted_fraud_rate:.4f}\n")
    fh.write(f"Flagged campaigns: {n_campaigns}\n")
    fh.write(f"Reviews covered by flagged campaigns: {reviews_in_campaigns}\n")
    fh.write(
        "\nThese four numbers (unweighted vs. review-weighted, all clusters vs. the 15\n"
        "largest) are what should be used to state whether large clusters skew genuine\n"
        "or fraudulent - not the unweighted multi-cluster average alone.\n"
    )

spark.stop()
