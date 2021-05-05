import pandas as pd
import numpy as np
from tqdm import tqdm
import os

NPY_DIR = ""

for set_ in ["train", "valid", "holdout"]:
    df = pd.read_csv(f"csv/brain-mri-abnormalness-{set_}-v2.csv")
    p = np.load(f"outputs/efficientnet2d_{set_}_embeddings.npy")
    imageuid = df["imageUid"].values
    numpies = {}
    for image_name, embedding_value in zip(imageuid, p):
        numpies[image_name] = embedding_value
    for s in tqdm(df["studyUid"].unique()):
        sub_df = df[df["studyUid"] == s]
        series_len = len(sub_df)
        sequence_len = 11
        for i in range(series_len):
            slices = []
            img_loc = sub_df.iloc[i,-1]
            img_name = sub_df.iloc[i,2]
            for ii in range(1, int((sequence_len + 1) / 2)):
                slices.append(numpies[sub_df[sub_df['loc'] == max(img_loc-ii,0)]["imageUid"].values[0]])
                slices = slices[::-1]
                slices.append(numpies[sub_df[sub_df['loc'] == img_loc]["imageUid"].values[0]])
            for ii in range(1, int((sequence_len + 1) / 2)):
                slices.append(numpies[sub_df[sub_df['loc'] == min(img_loc+ii,series_len-1)]["imageUid"].values[0]])
            slices = np.vstack(slices)
            np.save(os.path.join(NPY_DIR, img_name), slices)