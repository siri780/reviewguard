"""M12 - Text-only baseline on the Ott et al. deceptive opinion spam corpus (1,600 hotel reviews:
800 truthful, 800 deceptive written by paid crowd workers). TF-IDF over unigrams+bigrams ->
logistic regression (Spark MLlib), 5-fold CV using the corpus's own folds. No hyperparameter
tuning (regParam fixed at 0.01). Different domain/labels from YelpChi; NOT comparable to M1/M3."""
import glob, os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, SQLTransformer, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

rows = []
for p in glob.glob("data/optspam/op_spam_v1.4/*/*/*/*.txt"):
    parts = p.split(os.sep)
    fname = parts[-1]
    rows.append(dict(file=fname, polarity=parts[-4].replace("_polarity", ""), source=parts[-3],
                     fold=int(parts[-2].replace("fold", "")), hotel=fname.split("_")[1],
                     label=1 if parts[-3].startswith("deceptive") else 0,
                     text=open(p, encoding="utf-8", errors="ignore").read().strip()))
df = pd.DataFrame(rows).sort_values(["polarity", "source", "fold", "file"]).reset_index(drop=True)
print(len(df), "reviews;", df.label.mean(), "deceptive rate")

spark = SparkSession.builder.appName("ReviewGuard-M12").master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8").getOrCreate()
sdf = spark.createDataFrame(df)

def pipeline():
    return Pipeline(stages=[
        RegexTokenizer(inputCol="text", outputCol="uni", pattern="\\W+", toLowercase=True),
        NGram(n=2, inputCol="uni", outputCol="bi"),
        SQLTransformer(statement="SELECT *, concat(uni, bi) AS toks FROM __THIS__"),
        CountVectorizer(inputCol="toks", outputCol="tf", vocabSize=30000, minDF=2),
        IDF(inputCol="tf", outputCol="features"),
        LogisticRegression(featuresCol="features", labelCol="label", regParam=0.01, maxIter=100),
    ])

oof, per_fold = [], []
for f in range(1, 6):
    model = pipeline().fit(sdf.filter(sdf.fold != f))
    pred = model.transform(sdf.filter(sdf.fold == f)) \
        .select("file", "polarity", "hotel", "fold", "label", "prediction",
                vector_to_array("probability")[1].alias("prob")).toPandas()
    oof.append(pred)
    per_fold.append((f, accuracy_score(pred.label, pred.prediction), roc_auc_score(pred.label, pred.prob),
                     f1_score(pred.label, pred.prediction)))
    print("fold", f, "acc %.4f auc %.4f f1 %.4f" % per_fold[-1][1:])
oof = pd.concat(oof)
oof.to_csv("results/m12_oof_predictions.csv", index=False)

arr = np.array([r[1:] for r in per_fold])
lines = ["M12 Text-only baseline (Ott et al. corpus, TF-IDF uni+bigram LR, 5-fold CV, no tuning)\n"]
for f, a, u, s in per_fold:
    lines.append(f"fold {f}: accuracy {a:.4f}  ROC-AUC {u:.4f}  F1 {s:.4f}")
lines.append(f"mean +/- sd: accuracy {arr[:,0].mean():.4f} +/- {arr[:,0].std(ddof=1):.4f}  "
             f"ROC-AUC {arr[:,1].mean():.4f} +/- {arr[:,1].std(ddof=1):.4f}  F1 {arr[:,2].mean():.4f} +/- {arr[:,2].std(ddof=1):.4f}")
for pol in ["positive", "negative"]:
    s = oof[oof.polarity == pol]
    lines.append(f"{pol} polarity (out-of-fold): accuracy {accuracy_score(s.label, s.prediction):.4f}  ROC-AUC {roc_auc_score(s.label, s.prob):.4f}")
full = pipeline().fit(sdf)
vocab, coef = full.stages[3].vocabulary, full.stages[-1].coefficients.toArray()
order = np.argsort(coef)
lines.append("top terms -> deceptive: " + ", ".join(vocab[i] for i in order[::-1][:12]))
lines.append("top terms -> truthful:  " + ", ".join(vocab[i] for i in order[:12]))
print("\n".join(lines))
open("results/m12_text_baseline_results.txt", "w").write("\n".join(lines) + "\n")
spark.stop()
