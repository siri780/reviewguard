"""M15 - Boosting on the reviewer-disjoint split, with reviewer-cluster bootstrap CIs.

This is the 'next experiment' named in the report's conclusion. It needs only
results/_m3_features.csv (written by M3), results/_shared_split_row_ids.csv
(written by M1) and data/YelpChi.mat; no Spark. Run from the repo root:
    python src/m15_boosted_disjoint_ci.py

For each split (random shared split; reviewer-disjoint split built exactly as in M9)
and each learner (HistGradientBoosting, random_state=0, no tuning; scikit-learn
logistic regression with StandardScaler fit on train, max_iter=100) it fits three
feature sets (M1, M1+deg_rur, full M3) and reports test ROC-AUC/AUPRC.
It then resamples whole R-U-R reviewer clusters of the test set with replacement
(1,000 replicates, seed 0) and reports the paired AUC difference with a 95%
percentile interval. Single fit per model: the intervals reflect test-set
sampling, not training randomness. The sklearn logistic regression is a
cross-check of the Spark model, not a replacement for it."""
import numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import common

B, SEED = 1000, 0
mat, _, labels = common.load_mat()
n = len(labels)
ncomp, comp = connected_components(mat["net_rur"], directed=False)
df = pd.read_csv(common.M3_FEATURES_PATH)
df["comp"] = comp[df["row_id"].values]
split = pd.read_csv(common.SPLIT_PATH)
df = df.merge(split, on="row_id")

# reviewer-disjoint split: identical construction to M9
sizes = np.bincount(comp, minlength=ncomp)
rng = np.random.RandomState(42)
test_comps, total = set(), 0
for c in rng.permutation(ncomp):
    if total >= 0.2 * n:
        break
    test_comps.add(c)
    total += sizes[c]
disjoint_test = df["comp"].isin(test_comps)

cols = [f"f{i}" for i in range(32)]
sets = {"M1": cols, "M1+deg_rur": cols + ["deg_rur"],
        "M3": cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]}
splits = {"random shared split": df["split"] == "test",
          "reviewer-disjoint split": disjoint_test}


def fit_scores(kind, fc, tr, te):
    if kind == "boosting":
        m = HistGradientBoostingClassifier(random_state=0).fit(tr[fc], tr["label"])
        return m.predict_proba(te[fc])[:, 1]
    sc = StandardScaler().fit(tr[fc])
    m = LogisticRegression(max_iter=100).fit(sc.transform(tr[fc]), tr["label"])
    return m.predict_proba(sc.transform(te[fc]))[:, 1]


def boot_diffs(y, comps, scores, pairs):
    codes, uniq = pd.factorize(comps)
    idx_by = [np.where(codes == k)[0] for k in range(len(uniq))]
    r = np.random.RandomState(SEED)
    res = {p: [] for p in pairs}
    for _ in range(B):
        pick = r.randint(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by[k] for k in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        a = {k: roc_auc_score(yy, v[ii]) for k, v in scores.items()}
        for p in pairs:
            res[p].append(a[p[1]] - a[p[0]])
    return {p: (np.mean(v), *np.percentile(v, [2.5, 97.5])) for p, v in res.items()}


pairs = [("M1", "M1+deg_rur"), ("M1+deg_rur", "M3"), ("M1", "M3")]
out = ["M15 Boosting and logistic regression on both splits, with reviewer-cluster bootstrap 95% intervals",
       f"Bootstrap: {B} replicates over whole R-U-R clusters of the test set, seed {SEED}. Single model fit per cell.\n"]
for sname, is_test in splits.items():
    tr, te = df[~is_test], df[is_test]
    out.append(f"##### {sname}: train {len(tr)} / test {len(te)}; test fraud rate {te['label'].mean():.4f}")
    for kind in ("boosting", "logistic regression (scikit-learn)"):
        k = "boosting" if kind == "boosting" else "lr"
        scores = {nm: fit_scores(k, fc, tr, te) for nm, fc in sets.items()}
        y = te["label"].values
        out.append(f"-- {kind}")
        for nm, s in scores.items():
            out.append(f"   {nm:11s} AUC {roc_auc_score(y, s):.4f}  AUPRC {average_precision_score(y, s):.4f}")
        d = boot_diffs(y, te["comp"].values, scores, pairs)
        for (a, b), (mu, lo, hi) in d.items():
            point = roc_auc_score(y, scores[b]) - roc_auc_score(y, scores[a])
            out.append(f"   AUC gain {a} -> {b}: {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    out.append("")
# Seed sensitivity: HistGradientBoosting early-stops on a random validation split,
# so AUC moves with random_state. Report the M1 -> M3 gain over 10 seeds.
out.append("##### Boosting seed sensitivity (random_state 0-9; early stopping uses a random validation split)")
for sname, is_test in splits.items():
    tr, te = df[~is_test], df[is_test]
    rows = []
    for seed in range(10):
        a = {}
        for nm, fc in sets.items():
            m = HistGradientBoostingClassifier(random_state=seed).fit(tr[fc], tr["label"])
            a[nm] = roc_auc_score(te["label"], m.predict_proba(te[fc])[:, 1])
        rows.append(a)
    r = pd.DataFrame(rows)
    g = r["M3"] - r["M1"]
    out.append(f"{sname}: AUC mean (sd) M1 {r['M1'].mean():.4f} ({r['M1'].std():.4f}), "
               f"M1+deg_rur {r['M1+deg_rur'].mean():.4f} ({r['M1+deg_rur'].std():.4f}), "
               f"M3 {r['M3'].mean():.4f} ({r['M3'].std():.4f}); "
               f"M1->M3 gain mean {g.mean():+.4f}, range [{g.min():+.4f}, {g.max():+.4f}]")
out.append("")
txt = "\n".join(out)
print(txt)
open("results/m15_boosted_disjoint_ci_results.txt", "w").write(txt + "\n")
