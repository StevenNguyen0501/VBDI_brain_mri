import numpy as np
from sklearn.model_selection import GroupKFold
import pandas as pd

df = pd.read_csv("csv/brain-mri-abnormalness-05042021.csv")

X = df["imageUid"].values
y = df["abnormal"].values
groups = df["studyUid"].values
group_kfold = GroupKFold(n_splits=6)
group_kfold.get_n_splits(X, y, groups)
print(group_kfold)
for train_index, test_index in group_kfold.split(X, y, groups):
    break
holdout_csv = df.iloc[test_index]
df = df.iloc[train_index]


X = df["imageUid"].values
y = df["abnormal"].values
groups = df["studyUid"].values
group_kfold = GroupKFold(n_splits=5)
group_kfold.get_n_splits(X, y, groups)
print(group_kfold)
for train_index, test_index in group_kfold.split(X, y, groups):
    break
val_csv = df.iloc[test_index]
df = df.iloc[train_index]

df.to_csv("csv/brain-mri-abnormalness-train-v2.csv", index=False)
val_csv.to_csv("csv/brain-mri-abnormalness-valid-v2.csv", index=False)
holdout_csv.to_csv("csv/brain-mri-abnormalness-holdout-v2.csv", index=False)
