"""
ReviewGuard - M6 SHAP Explainability
Builds the same M3 feature set (32 handcrafted + degree x3 + PageRank),
trains an equivalent sklearn Logistic Regression, and explains it with
SHAP's exact LinearExplainer. Reports:
  - Fidelity: SHAP values exactly reconstruct the model's linear margin (exact for linear models)
  - Top features by mean |SHAP|
  - Stability: top-5 feature ranking consistency across 3 bootstrap resamples
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap

spark = SparkSession.builder.appName("ReviewGuard-M6-SHAP") \
    .master("local[*]") \
    .config("spark.jars.packages", "io.graphframes:graphframes-spark3_2.12:0.12.3") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf_checkpoints")

# ---- 1. Rebuild the M3 feature set ----
mat = sio.loadmat("data/YelpChi.mat")
features = mat["features"].toarray()
labels = mat["label"].flatten()
n = features.shape[0]
cols = [f"f{i}" for i in range(features.shape[1])]

def edges_from_sparse(mat_sparse):
    coo = mat_sparse.tocoo()
    mask = coo.row < coo.col
    return pd.DataFrame({"src": coo.row[mask], "dst": coo.col[mask]})

vertices = spark.createDataFrame(pd.DataFrame({"id": np.arange(n)}))

def degree_feature(relation_key, name):
    e = spark.createDataFrame(edges_from_sparse(mat[relation_key]))
    g = GraphFrame(vertices, e)
    return g.degrees.withColumnRenamed("degree", name)

deg_rur = degree_feature("net_rur", "deg_rur")
deg_rsr = degree_feature("net_rsr", "deg_rsr")
deg_rtr = degree_feature("net_rtr", "deg_rtr")

e_homo = spark.createDataFrame(edges_from_sparse(mat["homo"]))
g_homo = GraphFrame(vertices, e_homo)
pr = g_homo.pageRank(resetProbability=0.15, maxIter=5)
pagerank_df = pr.vertices.select("id", "pagerank")

graph_pdf = deg_rur.join(deg_rsr, "id").join(deg_rtr, "id").join(pagerank_df, "id").toPandas()
graph_pdf = graph_pdf.set_index("id").reindex(range(n)).fillna(0)

pdf = pd.DataFrame(features, columns=cols)
pdf["deg_rur"] = graph_pdf["deg_rur"].values
pdf["deg_rsr"] = graph_pdf["deg_rsr"].values
pdf["deg_rtr"] = graph_pdf["deg_rtr"].values
pdf["pagerank"] = graph_pdf["pagerank"].values
pdf["label"] = labels.astype(int)

feature_cols = cols + ["deg_rur", "deg_rsr", "deg_rtr", "pagerank"]
spark.stop()

# ---- 2. Train an sklearn LR equivalent to the Spark M3 model ----
X = pdf[feature_cols].values
y = pdf["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_s, y_train)
test_auc = clf.score(X_test_s, y_test)
print(f"sklearn LR test accuracy: {test_auc:.4f}")

# ---- 3. SHAP explainability (exact LinearExplainer) ----
explainer = shap.LinearExplainer(clf, X_train_s)
shap_values = explainer.shap_values(X_test_s)

# Fidelity: SHAP values + base value should reconstruct the model's margin exactly (linear model)
margin_from_model = clf.decision_function(X_test_s)
margin_from_shap = shap_values.sum(axis=1) + explainer.expected_value
fidelity_max_err = np.max(np.abs(margin_from_model - margin_from_shap))
print(f"Fidelity check - max abs error (SHAP sum vs model margin): {fidelity_max_err:.8f} (expect ~0)")

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_idx = np.argsort(mean_abs_shap)[::-1][:10]
print("\nTop 10 features by mean |SHAP value|:")
top_features_report = []
for i in top_idx:
    line = f"  {feature_cols[i]}: {mean_abs_shap[i]:.4f}"
    print(line)
    top_features_report.append(line)

# ---- 4. Stability: top-5 feature ranking across 3 bootstrap resamples ----
rankings = []
rng = np.random.RandomState(0)
for b in range(3):
    idx = rng.choice(len(X_train_s), size=len(X_train_s), replace=True)
    clf_b = LogisticRegression(max_iter=1000)
    clf_b.fit(X_train_s[idx], y_train[idx])
    exp_b = shap.LinearExplainer(clf_b, X_train_s[idx])
    sv_b = exp_b.shap_values(X_test_s)
    mean_abs_b = np.abs(sv_b).mean(axis=0)
    top5_b = set(np.argsort(mean_abs_b)[::-1][:5])
    rankings.append(top5_b)

overlap_01 = len(rankings[0] & rankings[1])
overlap_02 = len(rankings[0] & rankings[2])
overlap_12 = len(rankings[1] & rankings[2])
avg_overlap = (overlap_01 + overlap_02 + overlap_12) / 3
print(f"\nStability check - avg top-5 feature overlap across 3 bootstrap resamples: {avg_overlap:.2f} / 5")

with open("results/m6_shap_results.txt", "w") as fh:
    fh.write("M6 SHAP Explainability (LinearExplainer on sklearn LR, M3 feature set)\n")
    fh.write(f"sklearn LR test accuracy: {test_auc:.4f}\n")
    fh.write(f"Fidelity (max abs error, SHAP sum vs model margin): {fidelity_max_err:.8f}\n\n")
    fh.write("Top 10 features by mean |SHAP value|:\n")
    fh.write("\n".join(top_features_report))
    fh.write(f"\n\nStability - avg top-5 feature overlap across 3 bootstrap resamples: {avg_overlap:.2f} / 5\n")

print("\nSaved to results/m6_shap_results.txt")
