"""M11 - Synthetic scale-up. The YelpChi graph (features, R-U-R edges, labels) is replicated
k times as disjoint copies so row counts reach ~1M. Measures, per size: R-U-R degree,
connected components (GraphFrames), and logistic-regression training. SYNTHETIC workload:
it tests pipeline scaling, not detection quality. Cores via env var CORES (default: all).
Optional env vars: KS (copies, default 1,2,5,11,22), TRIALS (default 2), DRIVER_MEM (default 5g), SUFFIX (appended to the
output file name so a new run does not overwrite an earlier one; default empty). K=218 gives about 10.0 million rows."""
import os, time, json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
import common

CORES = os.environ.get("CORES", "*")
KS = [int(x) for x in os.environ.get("KS", "1,2,5,11,22").split(",")]
TRIALS = int(os.environ.get("TRIALS", "2"))
DRIVER_MEM = os.environ.get("DRIVER_MEM", "5g")
SUFFIX = os.environ.get("SUFFIX", "")
spark = SparkSession.builder.appName("ReviewGuard-M11").master(f"local[{CORES}]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.driver.memory", DRIVER_MEM).config("spark.sql.shuffle.partitions", "16").getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")
from graphframes import GraphFrame

mat, features, labels = common.load_mat()
n = len(labels)
cols = [f"f{i}" for i in range(32)]
pdf = pd.DataFrame(features, columns=cols); pdf["label"] = labels.astype(int); pdf["id"] = np.arange(n)
base_v = spark.createDataFrame(pdf)
base_e = spark.createDataFrame(common.edges_from_sparse(mat["net_rur"], bidirectional=False))

def build(k):
    reps = spark.range(k).withColumnRenamed("id", "copy")
    v = base_v.crossJoin(reps).withColumn("id", F.col("id") + F.col("copy") * n).drop("copy")
    e = base_e.crossJoin(reps).withColumn("src", F.col("src") + F.col("copy") * n) \
        .withColumn("dst", F.col("dst") + F.col("copy") * n).drop("copy")
    return v.cache(), e.cache()

def timed(fn):
    t = time.perf_counter(); r = fn(); return time.perf_counter() - t, r

def run(k):
    v, e = build(k)
    rows = v.count(); e.count()
    t_deg, deg = timed(lambda: (lambda d: (d.count(), d)[1])(
        e.select(F.col("src").alias("id")).union(e.select(F.col("dst").alias("id")))
         .groupBy("id").count().withColumnRenamed("count", "deg_rur")))
    t_cc, _ = timed(lambda: GraphFrame(v.select("id"), e).connectedComponents().count())
    full = v.join(deg, "id", "left").fillna(0, subset=["deg_rur"])
    vec = VectorAssembler(inputCols=cols + ["deg_rur"], outputCol="raw").transform(full)
    vec = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(vec).transform(vec).cache()
    vec.count()
    t_lr, _ = timed(lambda: LogisticRegression(featuresCol="features", labelCol="label", maxIter=20).fit(vec))
    v.unpersist(); e.unpersist(); vec.unpersist()
    return rows, t_deg, t_cc, t_lr

run(1)  # warm-up (JVM, JIT, GraphFrames JAR); discarded
res = []
for k in KS:
    for trial in range(TRIALS):
        rows, d, c, l = run(k)
        rec = dict(k=k, rows=rows, trial=trial, deg_s=round(d, 2), cc_s=round(c, 2), lr_s=round(l, 2))
        print(rec, flush=True); res.append(rec)
tag = CORES.replace("*", "all")
pd.DataFrame(res).to_csv(f"results/m11_scale_cores{tag}{SUFFIX}.csv", index=False)
print(f"saved results/m11_scale_cores{tag}{SUFFIX}.csv")
spark.stop()
