"""M4b - Label-free campaign flagging rule. Score = z(R-U-R cluster size) +
z(deg_rtr, burst proxy) + z(deg_rsr, rating-conformity proxy). No label used
in scoring; label used only afterward to evaluate (retrospective diagnostic)."""
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import roc_auc_score, average_precision_score
import common

spark = SparkSession.builder.appName("ReviewGuard-M4b") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")
from graphframes import GraphFrame

mat, features, labels = common.load_mat()
n = len(labels)
edges_rur = common.edges_from_sparse(mat["net_rur"], bidirectional=False)
vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n), "label": labels.astype(int)}))
g = GraphFrame(vertices, spark.createDataFrame(edges_rur))
cc = g.connectedComponents()
comp_size = cc.groupBy("component").agg(F.count("*").alias("comp_size"))
per_row = cc.join(comp_size, "component", "left").select("id", "comp_size")

m3 = pd.read_csv(common.M3_FEATURES_PATH)[["row_id", "deg_rtr", "deg_rsr", "label"]].rename(columns={"row_id": "id"})
joined = per_row.join(spark.createDataFrame(m3), "id", "inner").toPandas()
assert len(joined) == n

z = lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9)
joined["score"] = z(joined["comp_size"]) + z(joined["deg_rtr"]) + z(joined["deg_rsr"])

auc = roc_auc_score(joined["label"], joined["score"])
auprc = average_precision_score(joined["label"], joined["score"])
ranked = joined.sort_values("score", ascending=False).reset_index(drop=True)
def p_at(k):
    h = int(ranked.iloc[:k]["label"].sum()); return h / k, h
p548, h548 = p_at(548)
p1246, h1246 = p_at(1246)
base = float(labels.mean())

out = (f"M4b Label-Free Campaign Flagging Rule\n"
       f"Score = z(comp_size) + z(deg_rtr) + z(deg_rsr); no label used in scoring.\n\n"
       f"ROC-AUC vs label: {auc:.4f}\nAUPRC vs label: {auprc:.4f} (base rate {base:.4f})\n"
       f"Precision@548:  {p548:.4f} ({h548}/548)\nPrecision@1246: {p1246:.4f} ({h1246}/1246)\n")
print(out)
joined[["id", "comp_size", "deg_rtr", "deg_rsr", "score", "label"]].to_csv("results/_m4b_scores.csv", index=False)
open("results/m4b_label_free_results.txt", "w").write(out)
spark.stop()
