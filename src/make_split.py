"""
ReviewGuard - Shared Train/Test Split
Builds ONE stratified, deterministic split by row_id (outside Spark, so it's
immune to randomSplit's partition/order-dependence). Every other script joins
this file in instead of calling randomSplit itself, guaranteeing identical
train/test membership across M1, M3, M6.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy.io as sio

mat = sio.loadmat("data/YelpChi.mat")
labels = mat["label"].flatten()
n = len(labels)
row_ids = np.arange(n)

train_ids, test_ids = train_test_split(
    row_ids, test_size=0.2, random_state=42, stratify=labels
)

split_df = pd.DataFrame({"row_id": row_ids})
train_set = set(train_ids.tolist())
split_df["split"] = split_df["row_id"].apply(lambda x: "train" if x in train_set else "test")
split_df.to_csv("data/split.csv", index=False)

print(f"Train: {(split_df.split=='train').sum()}  Test: {(split_df.split=='test').sum()}")
print(f"Train fraud rate: {labels[train_ids].mean():.4f}")
print(f"Test fraud rate: {labels[test_ids].mean():.4f}")
print("Saved to data/split.csv")
