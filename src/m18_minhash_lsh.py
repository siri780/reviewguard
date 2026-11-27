"""M18 - MinHash locality-sensitive hashing for near-duplicate review search (Spark MLlib MinHashLSH)
on the Ott et al. corpus (1,600 hotel reviews). Run from the repo root with the corpus under
data/optspam/op_spam_v1.4/ (same layout as M12):
    python src/m18_minhash_lsh.py
Each review becomes the set of its lowercase word 3-grams. Spark's MinHashLSH (NUM_TABLES hash
tables, seed 42) proposes candidate pairs and returns each candidate's exact Jaccard distance;
pairs with Jaccard similarity >= 0.5 are kept. As a check, exact Jaccard similarity is computed
for all 1,279,200 pairs and LSH recall is reported at three similarity thresholds. We also report
how many LSH-found pairs share a label and a fold, and compare against the M12 cosine rule
(TF-IDF cosine >= 0.9). This is a scalable near-duplicate detector demonstrated on a small corpus;
it is not a detector of deceptive reviews."""
import glob, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, CountVectorizer, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NUM_TABLES = int(os.environ.get("NUM_TABLES", "20"))
rows = []
for p in glob.glob("data/optspam/op_spam_v1.4/*/*/*/*.txt"):
    parts = p.split(os.sep)
    rows.append(dict(file=parts[-1], polarity=parts[-4].replace("_polarity", ""), source=parts[-3],
                     fold=int(parts[-2].replace("fold", "")),
                     label=1 if parts[-3].startswith("deceptive") else 0,
                     text=open(p, encoding="utf-8", errors="ignore").read().strip()))
df = pd.DataFrame(rows).sort_values(["polarity", "source", "fold", "file"]).reset_index(drop=True)
df["id"] = np.arange(len(df))
n = len(df)
print(n, "reviews")

spark = SparkSession.builder.appName("ReviewGuard-M18").master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8").getOrCreate()
sdf = spark.createDataFrame(df[["id", "text"]])
prep = Pipeline(stages=[
    RegexTokenizer(inputCol="text", outputCol="uni", pattern="\\W+", toLowercase=True),
    NGram(n=3, inputCol="uni", outputCol="sh"),
    CountVectorizer(inputCol="sh", outputCol="vec", binary=True, vocabSize=1 << 20, minDF=1),
]).fit(sdf)
vec = prep.transform(sdf).filter(F.size("sh") > 0).select("id", "vec").cache()
lsh = MinHashLSH(inputCol="vec", outputCol="hashes", numHashTables=NUM_TABLES, seed=42).fit(vec)
joined = lsh.approxSimilarityJoin(vec, vec, 0.5, distCol="jdist") \
    .select(F.col("datasetA.id").alias("a"), F.col("datasetB.id").alias("b"), "jdist") \
    .filter("a < b").toPandas()
joined["sim"] = 1 - joined.jdist
print("LSH pairs with Jaccard >= 0.5:", len(joined))

# exact Jaccard for every pair, from the same shingle sets
V = vec.toPandas()
size = V.vec.iloc[0].size
M = sp.csr_matrix((np.ones(sum(len(v.indices) for v in V.vec)), np.concatenate([v.indices for v in V.vec]),
                   np.cumsum([0] + [len(v.indices) for v in V.vec])), shape=(len(V), size))
ids = V.id.values
inter = (M @ M.T).toarray().astype(float)
sz = np.asarray(M.sum(1)).ravel()
union = sz[:, None] + sz[None, :] - inter
J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
np.fill_diagonal(J, 0)
iu = np.triu_indices(len(V), 1)
ex = pd.DataFrame(dict(a=ids[iu[0]], b=ids[iu[1]], sim=J[iu]))
found = set(zip(joined.a, joined.b))
lines = ["M18 MinHash LSH near-duplicate search (Ott corpus, word 3-gram shingles)", "",
         f"Reviews: {n}; candidate pairs if compared exhaustively: {n*(n-1)//2:,}",
         f"MinHashLSH: {NUM_TABLES} hash tables, seed 42, join threshold Jaccard similarity >= 0.5; pairs returned: {len(joined)}",
         "Recall against exact Jaccard (all pairs):"]
for t in (0.9, 0.7, 0.5):
    e = ex[ex.sim >= t]
    hit = sum((a, b) in found for a, b in zip(e.a, e.b))
    exp_rec = 1 - (1 - t) ** NUM_TABLES
    lines.append(f"  similarity >= {t}: exact pairs {len(e)}, found by LSH {hit}"
                 + (f" (recall {hit/len(e):.3f}; expected for a pair exactly at {t}: {exp_rec:.3f})" if len(e) else " (no exact pairs)"))
lines.append("Pairs returned by LSH have exact Jaccard >= 0.5 by construction, so precision at 0.5 is 1.0 (Spark returns the exact distance).")
top = joined.sort_values("sim", ascending=False).head(10)
d = df.set_index("id")
if len(joined):
    same_label = np.mean([d.label[a] == d.label[b] for a, b in zip(joined.a, joined.b)])
    same_fold = np.mean([d.fold[a] == d.fold[b] for a, b in zip(joined.a, joined.b)])
    lines += [f"Share of LSH pairs with the same label: {same_label:.3f}; same fold: {same_fold:.3f}"]
# connected groups at >= 0.5
parent = list(range(n))
def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]; i = parent[i]
    return i
for a, b in zip(joined.a, joined.b):
    parent[find(a)] = find(b)
g = pd.Series([find(i) for i in range(n)])
lines.append(f"Reviews -> groups after linking LSH pairs (Jaccard >= 0.5): {n} -> {g.nunique()}")
# comparison with the M12 rule
S = cosine_similarity(TfidfVectorizer().fit_transform(df.text)); np.fill_diagonal(S, 0)
ii, jj = np.where(np.triu(S) >= 0.9)
cos_pairs = set(zip(ii.tolist(), jj.tolist()))
lines.append(f"M12 cosine rule (TF-IDF cosine >= 0.9): {len(cos_pairs)} pairs; also found by LSH: {len(cos_pairs & found)}")
lines.append("Most similar pairs (file names, Jaccard):")
for a, b, s in zip(top.a, top.b, top.sim):
    lines.append(f"  {d.file[a]} | {d.file[b]} | {s:.3f}")
txt = "\n".join(lines)
print(txt)
os.makedirs("results", exist_ok=True)
open("results/m18_minhash_lsh_results.txt", "w").write(txt + "\n")
joined.assign(file_a=[d.file[a] for a in joined.a], file_b=[d.file[b] for b in joined.b]) \
    .sort_values("sim", ascending=False).to_csv("results/m18_lsh_pairs.csv", index=False)
spark.stop()
