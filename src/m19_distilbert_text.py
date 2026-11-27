"""M19 - DistilBERT fine-tuning on the Ott et al. corpus (1,600 hotel reviews), evaluated on the same
duplicate-grouped five folds as the TF-IDF baseline (M12g). Run from the repo root with the corpus under
data/optspam/op_spam_v1.4/ (same layout as M12):
    python src/m19_distilbert_text.py
Each fold fine-tunes distilbert-base-uncased on the other four folds (about 1,280 reviews) and scores the
held-out fold; no hyperparameter search, one seed (42). Settings via environment variables:
    MODEL (default distilbert-base-uncased), MAXLEN (256), EPOCHS (3), BATCH (16), LR (5e-5),
    FOLDS (default 1,2,3,4,5).
Each finished fold is saved to results/m19_fold<k>.csv and skipped on re-runs, so the job can be stopped and
resumed. When all five folds exist, results/m19_distilbert_results.txt is written with per-fold accuracy,
ROC-AUC and F1 next to the TF-IDF logistic-regression values parsed from results/m12g_text_baseline_results.txt.
Different corpus and labels from YelpChi; the numbers are not comparable to M1/M3. Five folds and one seed
cannot establish that one text model is better than the other."""
import glob, os, re, time
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

MODEL = os.environ.get("MODEL", "distilbert-base-uncased")
MAXLEN = int(os.environ.get("MAXLEN", "256"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH = int(os.environ.get("BATCH", "16"))
LR = float(os.environ.get("LR", "5e-5"))
FOLDS = [int(x) for x in os.environ.get("FOLDS", "1,2,3,4,5").split(",")]
SEED = 42
torch.set_num_threads(os.cpu_count() or 2)

rows = []
for p in glob.glob("data/optspam/op_spam_v1.4/*/*/*/*.txt"):
    parts = p.split(os.sep)
    rows.append(dict(file=parts[-1], polarity=parts[-4].replace("_polarity", ""), source=parts[-3],
                     fold=int(parts[-2].replace("fold", "")),
                     label=1 if parts[-3].startswith("deceptive") else 0,
                     text=open(p, encoding="utf-8", errors="ignore").read().strip()))
df = pd.DataFrame(rows).sort_values(["polarity", "source", "fold", "file"]).reset_index(drop=True)
print(len(df), "reviews;", round(df.label.mean(), 3), "deceptive rate", flush=True)

# same duplicate grouping as M12g, so the folds are identical
norm = df.text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
parent = list(range(len(df)))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for g in norm.groupby(norm).groups.values():
    g = list(g)
    for j in g[1:]:
        parent[find(j)] = find(g[0])
S = cosine_similarity(TfidfVectorizer().fit_transform(df.text)); np.fill_diagonal(S, 0)
ii, jj = np.where(np.triu(S) >= 0.9)
for a, b in zip(ii, jj):
    parent[find(a)] = find(b)
df["group"] = [int(find(i)) for i in range(len(df))]
df["fold"] = df.groupby("group").fold.transform("min")
print(len(df), "reviews ->", df.group.nunique(), "groups", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
os.makedirs("results", exist_ok=True)

def batches(idx, enc_ids, labels, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), BATCH):
        b = idx[s:s + BATCH]
        pad = tok.pad({"input_ids": [enc_ids[i] for i in b]}, return_tensors="pt")
        yield pad["input_ids"], pad["attention_mask"], torch.tensor(labels[b])

enc = tok(df.text.tolist(), truncation=True, max_length=MAXLEN)["input_ids"]
lab = df.label.values
for f in FOLDS:
    out = f"results/m19_fold{f}.csv"
    if os.path.exists(out):
        print("fold", f, "already done; skipping", flush=True); continue
    torch.manual_seed(SEED); rng = np.random.RandomState(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    tr_idx, te_idx = df.index[df.fold != f].tolist(), df.index[df.fold == f].tolist()
    steps = EPOCHS * int(np.ceil(len(tr_idx) / BATCH))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; nb = 0
        for ids, mask, y in batches(tr_idx, enc, lab, True, rng):
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); tot += loss.item(); nb += 1
        print(f"fold {f} epoch {ep+1}/{EPOCHS} loss {tot/nb:.4f} elapsed {time.time()-t0:.0f}s", flush=True)
    model.eval(); probs = []
    order = sorted(te_idx, key=lambda i: len(enc[i]))
    with torch.no_grad():
        for ids, mask, y in batches(order, enc, lab, False, rng):
            probs += torch.softmax(model(input_ids=ids, attention_mask=mask).logits, -1)[:, 1].tolist()
    pred = pd.DataFrame(dict(file=df.file[order].values, polarity=df.polarity[order].values, fold=f,
                             label=lab[order], prob=probs))
    pred.to_csv(out, index=False)
    print(f"fold {f}: accuracy {accuracy_score(pred.label, pred.prob >= .5):.4f} "
          f"ROC-AUC {roc_auc_score(pred.label, pred.prob):.4f} F1 {f1_score(pred.label, pred.prob >= .5):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)

have = [f for f in range(1, 6) if os.path.exists(f"results/m19_fold{f}.csv")]
if len(have) == 5:
    base = {}
    if os.path.exists("results/m12g_text_baseline_results.txt"):
        for m in re.finditer(r"fold (\d): accuracy ([\d.]+)\s+ROC-AUC ([\d.]+)\s+F1 ([\d.]+)",
                             open("results/m12g_text_baseline_results.txt").read()):
            base[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    L = [f"M19 DistilBERT ({MODEL}) on the Ott corpus: max length {MAXLEN}, {EPOCHS} epochs, batch {BATCH}, lr {LR}, seed {SEED}, "
         "duplicate-grouped five folds identical to M12g, no tuning", ""]
    A = []
    for f in range(1, 6):
        p = pd.read_csv(f"results/m19_fold{f}.csv")
        a, u, s = accuracy_score(p.label, p.prob >= .5), roc_auc_score(p.label, p.prob), f1_score(p.label, p.prob >= .5)
        A.append((a, u, s))
        b = f"   TF-IDF LR: accuracy {base[f][0]:.4f} ROC-AUC {base[f][1]:.4f} F1 {base[f][2]:.4f}" if f in base else ""
        L.append(f"fold {f}: accuracy {a:.4f}  ROC-AUC {u:.4f}  F1 {s:.4f}{b}")
    A = np.array(A)
    L.append(f"mean +/- sd DistilBERT: accuracy {A[:,0].mean():.4f} +/- {A[:,0].std(ddof=1):.4f}  "
             f"ROC-AUC {A[:,1].mean():.4f} +/- {A[:,1].std(ddof=1):.4f}  F1 {A[:,2].mean():.4f} +/- {A[:,2].std(ddof=1):.4f}")
    if len(base) == 5:
        B = np.array([base[f] for f in range(1, 6)])
        L.append(f"mean +/- sd TF-IDF LR:  accuracy {B[:,0].mean():.4f} +/- {B[:,0].std(ddof=1):.4f}  "
                 f"ROC-AUC {B[:,1].mean():.4f} +/- {B[:,1].std(ddof=1):.4f}  F1 {B[:,2].mean():.4f} +/- {B[:,2].std(ddof=1):.4f}")
        d = A[:, 1] - B[:, 1]
        L.append(f"ROC-AUC difference (DistilBERT - TF-IDF) per fold: {', '.join(f'{x:+.4f}' for x in d)}; "
                 f"mean {d.mean():+.4f}; DistilBERT higher in {int((d > 0).sum())} of 5 folds")
    L.append("Five folds and one seed: this does not establish that either model is better.")
    txt = "\n".join(L); print(txt)
    open("results/m19_distilbert_results.txt", "w").write(txt + "\n")
else:
    print("Folds finished so far:", have, "- run again to finish the rest.")
