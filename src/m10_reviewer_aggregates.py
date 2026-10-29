"""M10 - M2 proxy: reviewer-level aggregate features. For each review, the mean of
the 32 handcrafted features over all reviews by the same reviewer (R-U-R component).
No label is used. Compared on the random shared split and the reviewer-disjoint split."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from pyspark.sql import SparkSession
import common

spark = SparkSession.builder.appName("ReviewGuard-M10").master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8").getOrCreate()
mat, features, labels = common.load_mat()
n = len(labels)
ncomp, comp = connected_components(mat["net_rur"], directed=False)
df = pd.read_csv(common.M3_FEATURES_PATH)
df["comp"] = comp[df["row_id"].values]
cols = [f"f{i}" for i in range(32)]
rm = df.groupby("comp")[cols].transform("mean").add_prefix("rm_")
df = pd.concat([df, rm], axis=1)
rm_cols = list(rm.columns)

# random shared split (same as M1/M3)
tr_r, te_r = common.stratified_split(df, reuse=True)

# reviewer-disjoint split (same construction as M9)
sizes = np.bincount(comp, minlength=ncomp)
rng = np.random.RandomState(42)
test_comps, total = set(), 0
for c in rng.permutation(ncomp):
    if total >= 0.2 * n:
        break
    test_comps.add(c); total += sizes[c]
is_test = df["comp"].isin(test_comps)
tr_d, te_d = df[~is_test].copy(), df[is_test].copy()

runs = {
    "M1 + deg_rur": cols + ["deg_rur"],
    "M1 + deg_rur + reviewer-mean features (M2 proxy)": cols + ["deg_rur"] + rm_cols,
    "M1 + reviewer-mean features (no deg_rur)": cols + rm_cols,
}
out = ["M10 Reviewer-aggregate (M2 proxy) features\n"]
for split, (tr, te) in {"random shared split": (tr_r, te_r), "reviewer-disjoint split": (tr_d, te_d)}.items():
    out.append(f"\n##### {split} (train {len(tr)}, test {len(te)})")
    print(f"\n##### {split}")
    for name, fc in runs.items():
        m = common.train_eval_lr(spark, fc, tr, te)
        line = (f"{name}: AUC {m['roc_auc']:.4f}  AUPRC {m['auprc']:.4f}  "
                f"fraud F1 {m['fraud_f1']:.4f}  recall {m['fraud_recall']:.4f}")
        print(line); out.append(line)
open("results/m10_reviewer_aggregates_results.txt", "w").write("\n".join(out) + "\n")
spark.stop()
