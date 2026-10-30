"""
ReviewGuard - shared utilities for a fair, leakage-aware M1 vs M3 comparison.

Why this module exists (fixes applied after review):
  1. M1 and M3 previously called `randomSplit([.8,.2], seed=42)` independently
     on DataFrames with different row orders/partitioning, which does NOT
     guarantee the same rows land in train/test for both models. This module
     computes ONE stratified split by row_id, persists it to
     results/_shared_split_row_ids.csv, and every script (M1, M3, the
     ablations, SHAP) loads that same file, so the M1->M3 comparison is on
     an identical held-out set.
  2. Both models previously reported only ROC-AUC and Spark's weighted F1,
     which hides fraud-class performance at a 14.5% base rate. `evaluate()`
     adds AUPRC, fraud-class precision/recall/F1, and a confusion matrix.
  3. Edge construction previously used `row < col` (one-way edges only),
     which under-counts degree-style features and starves PageRank (a
     directed algorithm) of rank mass if a node's only edges are outbound.
     `edges_from_sparse(..., bidirectional=True)` keeps both directions.
"""
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
)

SPLIT_PATH = "results/_shared_split_row_ids.csv"
M3_FEATURES_PATH = "results/_m3_features.csv"


def load_mat(path="data/YelpChi.mat"):
    mat = sio.loadmat(path)
    features = mat["features"].toarray()
    labels = mat["label"].flatten().astype(int)
    return mat, features, labels


def edges_from_sparse(mat_sparse, bidirectional=True):
    """Edge list (src, dst) from a scipy sparse adjacency matrix.

    bidirectional=True (default) keeps both (i,j) and (j,i): required for
    PageRank, since GraphFrames' pageRank treats edges as directed and a
    one-way edge list only lets rank mass flow in one direction across each
    pair. bidirectional=False keeps one row per undirected pair (i<j), which
    is sufficient and slightly cheaper for a plain undirected degree count.
    """
    coo = mat_sparse.tocoo()
    mask = coo.row != coo.col  # drop self-loops
    src, dst = coo.row[mask], coo.col[mask]
    if bidirectional:
        src2 = np.concatenate([src, dst])
        dst2 = np.concatenate([dst, src])
        return pd.DataFrame({"src": src2, "dst": dst2}).drop_duplicates().reset_index(drop=True)
    m2 = src < dst
    return pd.DataFrame({"src": src[m2], "dst": dst[m2]}).reset_index(drop=True)


def stratified_split(pdf, label_col="label", test_frac=0.2, seed=42, reuse=True):
    """Row-id-level stratified split, persisted so every script reuses the
    exact same train/test rows. Call with reuse=True everywhere except the
    very first script that creates it (M1), which should also use
    reuse=True once the file exists - it's idempotent either way: if the
    split file is missing it is created; if present it is loaded as-is."""
    os.makedirs("results", exist_ok=True)
    if reuse and os.path.exists(SPLIT_PATH):
        split_df = pd.read_csv(SPLIT_PATH)
        test_ids = set(split_df.loc[split_df["split"] == "test", "row_id"])
    else:
        rng = np.random.RandomState(seed)
        test_ids = set()
        for lbl in sorted(pdf[label_col].unique()):
            ids = pdf.loc[pdf[label_col] == lbl, "row_id"].values.copy()
            rng.shuffle(ids)
            n_test = int(round(len(ids) * test_frac))
            test_ids.update(ids[:n_test].tolist())
        split_df = pd.DataFrame({
            "row_id": pdf["row_id"],
            "split": ["test" if rid in test_ids else "train" for rid in pdf["row_id"]],
        })
        split_df.to_csv(SPLIT_PATH, index=False)
    is_test = pdf["row_id"].isin(test_ids)
    return pdf.loc[~is_test].copy(), pdf.loc[is_test].copy()


def evaluate(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "fraud_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fraud_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
    }


def format_results(header, metrics, extra_note=""):
    cm = metrics["confusion_matrix"]
    lines = [
        header,
        "",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        f"AUPRC:   {metrics['auprc']:.4f}",
        f"Fraud-class F1:        {metrics['fraud_f1']:.4f}",
        f"Fraud-class precision: {metrics['fraud_precision']:.4f}",
        f"Fraud-class recall:    {metrics['fraud_recall']:.4f}",
        f"Weighted F1 (reference only): {metrics['weighted_f1']:.4f}",
        f"Confusion matrix [[TN,FP],[FN,TP]]: {cm}",
    ]
    if extra_note:
        lines.append("")
        lines.append(extra_note)
    return "\n".join(lines) + "\n"


def train_eval_lr(spark, feature_cols, train_pdf, test_pdf, label_col="label"):
    """Fit scaled Logistic Regression on `feature_cols` and return the full
    metric dict from evaluate(). Shared by M1, M3, and every ablation run so
    the modeling procedure itself is never a hidden source of difference."""
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.functions import vector_to_array

    train_sdf = spark.createDataFrame(train_pdf[feature_cols + [label_col]])
    test_sdf = spark.createDataFrame(test_pdf[feature_cols + [label_col]])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    train_v = assembler.transform(train_sdf)
    test_v = assembler.transform(test_sdf)
    scaler_model = scaler.fit(train_v)
    train_s = scaler_model.transform(train_v)
    test_s = scaler_model.transform(test_v)

    lr = LogisticRegression(featuresCol="features", labelCol=label_col, maxIter=100)
    model = lr.fit(train_s)
    preds = model.transform(test_s)

    preds_pdf = preds.select(
        label_col, "prediction", vector_to_array("probability")[1].alias("prob1")
    ).toPandas()

    return evaluate(preds_pdf[label_col], preds_pdf["prediction"], preds_pdf["prob1"])
