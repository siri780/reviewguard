"""M9 - Reviewer-disjoint evaluation: whole R-U-R clusters go to train OR test,
never both. Compares M1, M1+deg_rur, and full M3 on this stricter split."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from pyspark.sql import SparkSession
import common

spark = SparkSession.builder.appName("ReviewGuard-M9").master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8").getOrCreate()
mat, features, labels = common.load_mat()
n = len(labels)
ncomp, comp = connected_components(mat["net_rur"], directed=False)
df = pd.read_csv(common.M3_FEATURES_PATH)
df["comp"] = comp[df["row_id"].values]
sizes = np.bincount(comp, minlength=ncomp)

rng = np.random.RandomState(42)
test_comps, total = set(), 0
for c in rng.permutation(ncomp):
    if total >= 0.2 * n:
        break
    test_comps.add(c)
    total += sizes[c]
is_test = df["comp"].isin(test_comps)
train, test = df[~is_test].copy(), df[is_test].copy()
assert set(train["comp"]).isdisjoint(set(test["comp"]))
print(f"Train {len(train)} / Test {len(test)}; test fraud rate {test['label'].mean():.4f}")

cols = [f"f{i}" for i in range(32)]
runs = {
    "M1 (handcrafted)": cols,
    "M1 + deg_rur": cols + ["deg_rur"],
    "M3 (full network)": cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"],
}
out = ["M9 Reviewer-disjoint split (whole R-U-R clusters held out)\n",
       f"Train {len(train)} / Test {len(test)}; test fraud rate {test['label'].mean():.4f}\n"]
for name, fc in runs.items():
    m = common.train_eval_lr(spark, fc, train, test)
    print(f"\n=== {name} ===")
    for k, v in m.items():
        print(f"{k}: {v}")
    out.append(common.format_results(f"\n=== {name} ===", m))
open("results/m9_reviewer_disjoint_results.txt", "w").write("\n".join(out))
spark.stop()
