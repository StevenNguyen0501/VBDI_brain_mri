import pandas as pd
import numpy as np
from tqdm import tqdm
import os

NPY_DIR = ""

for set_ in ["train", "valid", "holdout"]:
    df = pd.read_csv(f"csv/brain-mri-abnormalness-{set_}-v2.csv")
    p = np.load(f"outputs/efficientnet3d_{set_}_embeddings.npy")
    imageuid = df["imageUid"].values
    
    numpies = {}
    for image_name, embedding_value in zip(imageuid, p):
        numpies[image_name] = embedding_value
        
    for s in tqdm(df["studyUid"].unique()):
        sub_df = df[df["studyUid"] == s].sort_values(by=["loc"])
        series_len = len(sub_df)
        sequence_len = 31
        series_embedding = []
        series_embedding = [numpies[image_name] for image_name in sub_df["imageUid"].values]
        if series_len < sequence_len:
            pad = sequence_len - series_len
            pad_above = int(pad / 2)
            pad_below = pad - pad_above
            series_embedding = list(np.zeros(shape=(pad_above, 1280))) + series_embedding + list(np.zeros(shape=(pad_below, 1280)))
        series_embedding = np.vstack(series_embedding)
        np.save(os.path.join(NPY_DIR, s), series_embedding)

for set_ in ["train", "valid", "holdout"]:
    df = pd.read_csv(f"csv/brain-mri-abnormalness-{set_}-v2.csv")
    data = []
    for s in tqdm(df["studyUid"].unique()):    
        abnormal = df[df["studyUid"] == s]["abnormal"].max()
        data.append([s, abnormal])
    s_df = pd.DataFrame(data=data, columns=["imageUid", "abnormal"])
    s_df.to_csv(f"csv/brain-mri-abnormalness-{set_}-study.csv", index=False)        