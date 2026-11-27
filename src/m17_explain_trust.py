"""M17 - LIME vs SHAP on the boosted model, a reviewer-count counterfactual, and a calibrated Trust Score.

Needs results/_m3_features.csv, results/_shared_split_row_ids.csv (no Spark). Run from the repo root:
    python src/m17_explain_trust.py
Part A  Local explanations for the boosted model (HistGradientBoosting, random_state 0, 36 features).
        LIME (tabular, 2,000 perturbations) and Kernel SHAP (100 training background rows) explain the same
        60 random test reviews (RandomState 0). We report top-5 overlap and rank correlation of |attribution|.
Part B  Counterfactual for the logistic regression (scikit-learn copy of M3): for each test review flagged at
        0.5, the smallest reviewer count (deg_rur, other features fixed) that moves the score below 0.5,
        searched up to the 99th percentile of observed deg_rur.
Part C  Trust Score = 100 x (1 - calibrated probability that the review is filtered). Isotonic calibration is
        fit on a held-out 20% of the training rows; models are refit on the other 80%. We report Brier score,
        expected calibration error (10 bins) and the filtered share in each Trust-Score decile on the test set.
Everything is evaluated against Yelp's filter labels, not proven fraud."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import shap
from lime.lime_tabular import LimeTabularExplainer
import common

d = pd.read_csv(common.M3_FEATURES_PATH).merge(pd.read_csv(common.SPLIT_PATH), on="row_id")
tr, te = d[d.split == "train"], d[d.split == "test"].reset_index(drop=True)
cols = [f"f{i}" for i in range(32)] + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]
np.random.seed(0)
out = []

# ---------------- Part A
gb = HistGradientBoostingClassifier(random_state=0).fit(tr[cols], tr.label)
rs = np.random.RandomState(0)
idx = rs.choice(len(te), 60, replace=False)
Xte = te.loc[idx, cols].values
bg = tr[cols].sample(100, random_state=0).values
f = lambda X: gb.predict_proba(pd.DataFrame(X, columns=cols))[:, 1]
lime = LimeTabularExplainer(tr[cols].values, feature_names=cols, discretize_continuous=False, random_state=0, mode="classification")
shp = shap.KernelExplainer(f, bg)
sv = shp.shap_values(Xte, nsamples=300, silent=True)
lv = np.zeros_like(sv)
for i, x in enumerate(Xte):
    e = lime.explain_instance(x, lambda X: np.c_[1 - f(X), f(X)], num_features=len(cols), num_samples=2000)
    for j, w in e.as_map()[1]:
        lv[i, j] = w
ov, rho = [], []
for i in range(len(idx)):
    a, b = np.abs(sv[i]), np.abs(lv[i])
    ov.append(len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5])))
    rho.append(spearmanr(a, b).correlation)
gs, gl = np.abs(sv).mean(0), np.abs(lv).mean(0)
topg = lambda g: [cols[j] for j in np.argsort(-g)[:5]]
out += ["PART A  LIME vs Kernel SHAP on the boosted model (60 random test reviews)",
        f"Mean top-5 overlap per review: {np.mean(ov):.2f} / 5 (range {min(ov)}-{max(ov)})",
        f"Mean Spearman rank correlation of |attribution| per review: {np.nanmean(rho):.3f}",
        f"Global top-5 by mean |SHAP|: {topg(gs)}",
        f"Global top-5 by mean |LIME|: {topg(gl)}",
        f"Global rank correlation (36 features): {spearmanr(gs, gl).correlation:.3f}",
        f"deg_rur rank by mean |SHAP|: {int(np.argsort(np.argsort(-gs))[cols.index('deg_rur')]) + 1} of 36; by mean |LIME|: {int(np.argsort(np.argsort(-gl))[cols.index('deg_rur')]) + 1} of 36",
        "Attributions are for one boosted fit; LIME and Kernel SHAP are sampling-based approximations.", ""]

# ---------------- Part B
sc = StandardScaler().fit(tr[cols])
lr = LogisticRegression(max_iter=100).fit(sc.transform(tr[cols]), tr.label)
p = lr.predict_proba(sc.transform(te[cols]))[:, 1]
fl = np.where(p >= 0.5)[0]
cap = int(np.percentile(d.deg_rur, 99))
j = cols.index("deg_rur")
need = []
for i in fl:
    x = te.loc[i, cols].values.astype(float).copy()
    found = None
    for k in range(int(x[j]), cap + 1):
        x[j] = k
        if lr.predict_proba(sc.transform(pd.DataFrame([x], columns=cols)))[:, 1][0] < 0.5:
            found = k
            break
    need.append(found)
need_s = pd.Series(need, dtype="float")
flip = need_s.notna()
filt = te.loc[fl, "label"].values == 1
out += ["PART B  Reviewer-count counterfactual (logistic regression, threshold 0.5)",
        f"Flagged test reviews: {len(fl)} ({int(filt.sum())} filtered, {int((~filt).sum())} not filtered); the Spark M3 model flags 562 (375 + 187), so this scikit-learn copy differs slightly",
        f"Search range: observed reviewer count up to its 99th percentile ({cap}).",
        f"Flagged reviews that drop below 0.5 by raising only the reviewer count: {int(flip.sum())} ({100*flip.mean():.1f}%)",
        f"  among filtered: {100*flip[filt].mean():.1f}%; among not filtered: {100*flip[~filt].mean():.1f}%",
        f"Median reviewer count needed among those that flip: {need_s[flip].median():.0f} (current median {np.median(te.loc[fl,'deg_rur']):.0f})",
        f"Share that flip when the reviewer count is at most 1: {100*(need_s<=1).mean():.1f}%; at most 2: {100*(need_s<=2).mean():.1f}%",
        f"Share of flagged reviews whose current reviewer count is 0: {100*(te.loc[fl,'deg_rur']==0).mean():.1f}%",
        "This is a model-sensitivity statement, not advice on how a reviewer could evade detection.", ""]

# ---------------- Part C
tr2, cal = train_test_split(tr, test_size=0.2, random_state=0, stratify=tr.label)
def ece(y, p, n=10):
    b = np.minimum((p * n).astype(int), n - 1)
    return sum(abs(y[b == k].mean() - p[b == k].mean()) * (b == k).mean() for k in range(n) if (b == k).any())
res = {}
sc2 = StandardScaler().fit(tr2[cols])
models = {"logistic regression": (LogisticRegression(max_iter=100).fit(sc2.transform(tr2[cols]), tr2.label), lambda X: sc2.transform(X)),
          "boosting": (HistGradientBoostingClassifier(random_state=0).fit(tr2[cols], tr2.label), lambda X: X.values)}
out += ["PART C  Trust Score = 100 x (1 - calibrated probability of being filtered); test set 9,190 reviews"]
y = te.label.values
for name, (m, tf) in models.items():
    raw_c = m.predict_proba(tf(cal[cols]))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_c, cal.label)
    raw = m.predict_proba(tf(te[cols]))[:, 1]
    cp = iso.predict(raw)
    trust = 100 * (1 - cp)
    out.append(f"-- {name}: AUC {roc_auc_score(y, raw):.4f}; Brier raw {brier_score_loss(y, raw):.4f} -> calibrated {brier_score_loss(y, cp):.4f}; ECE raw {ece(y, raw):.4f} -> calibrated {ece(y, cp):.4f}")
    order = np.argsort(trust)
    dec = np.array_split(order, 10)
    out.append("   filtered share by Trust decile (lowest trust first): " + ", ".join(f"{100*y[k].mean():.1f}%" for k in dec))
    out.append(f"   lowest-trust decile holds {100*y[dec[0]].sum()/y.sum():.1f}% of all filtered reviews; base rate {100*y.mean():.2f}%")
out += ["Calibration is fit on 20% of the training rows. The Trust Score is a calibrated ranking of Yelp-filter risk; it is not validated against human judgment of trust."]
txt = "\n".join(out)
print(txt)
open("results/m17_explain_trust_results.txt", "w").write(txt + "\n")
