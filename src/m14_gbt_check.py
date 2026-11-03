"""M14 - Does the network gain survive a non-linear model? HistGradientBoosting on the same split."""
import glob, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

df = pd.read_csv("results/_m3_features.csv")
split = None
for p in glob.glob("results/*.csv") + glob.glob("data/*.csv"):
    h = pd.read_csv(p, nrows=2)
    if {"row_id", "split"} <= set(h.columns):
        split = pd.read_csv(p)[["row_id", "split"]]; print("split file:", p); break
assert split is not None, "no split file found"
df = df.merge(split, on="row_id")
base = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
configs = {"M1 (32 handcrafted)": base,
           "M1 + deg_rur": base + ["deg_rur"],
           "Full M3 (+4 graph)": base + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]}
tr, te = df[df.split == "train"], df[df.split == "test"]
out = ["M14 HistGradientBoosting on the shared split (no tuning, random_state=0)\n"]
for name, cols in configs.items():
    m = HistGradientBoostingClassifier(random_state=0).fit(tr[cols], tr["label"])
    p = m.predict_proba(te[cols])[:, 1]; y = te["label"]; pred = (p >= 0.5).astype(int)
    line = (f"{name}: AUC {roc_auc_score(y,p):.4f}  AUPRC {average_precision_score(y,p):.4f}  "
            f"fraud F1 {f1_score(y,pred):.4f}  precision {precision_score(y,pred):.4f}  recall {recall_score(y,pred):.4f}")
    print(line); out.append(line)
open("results/m14_gbt_results.txt", "w").write("\n".join(out) + "\n")
