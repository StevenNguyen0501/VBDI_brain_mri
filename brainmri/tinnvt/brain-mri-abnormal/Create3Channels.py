#%%
import os
import cv2
import tqdm
import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np
# %%
path_slice_copied = "./csv_new/brain-mri-xml-bboxes-copy-3.pkl"
path_sequence_path = "/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/df_series_clear.csv"

df_sequence = pd.read_csv(path_sequence_path)
df_sequence = df_sequence[(df_sequence["SeriesLabel"]=="FLAIR") | (df_sequence["SeriesLabel"]=="T1C") | (df_sequence["SeriesLabel"]=="T2")]
df_sequence

#%%%
df_copied = pd.read_pickle(path_slice_copied)
df_copied

#%%%
# Add sequence name for based image Uid
df_sequence_1 = df_sequence.rename(columns={"SeriesInstanceUID":"based_seriesUid", "StudyInstanceUID":"studyUid"})
df_copied = pd.merge(df_copied, df_sequence_1, on=["based_seriesUid", "studyUid"])
df_copied = df_copied.rename(columns={"SeriesLabel": "BasedSeriesLabel"})
df_copied

#%%%
df_sequence_2 = df_sequence.rename(columns={"SeriesInstanceUID":"copied_seriesUid", "StudyInstanceUID":"studyUid"})
df_copied = pd.merge(df_copied, df_sequence_2, on=["copied_seriesUid", "studyUid"])
df_copied = df_copied.rename(columns={"SeriesLabel": "CopiedSeriesLabel"})
df_copied


#%%%
df_copied["BasedSeriesLabel"].value_counts()

# %%
df_copied["CopiedSeriesLabel"].value_counts()

# %%
df_copied = df_copied.dropna()
df_copied.to_pickle("./csv_new/brain-mri-xml-bboxes-copy-final.pkl")
df_copied
#%%%
list(df_copied.columns)

# %%
# Visualization slices after copyings
IMG_DIR = "/home/single3/tintrung/images"
IMG_BOX_DIR = "/home/single3/tintrung/brain-mri-tumor-images-bboxes-final"
df = df_copied

def draw_bboxes_compare_two_sequences(image_based_name, image_copied_name):
    img1 = cv2.imread(f"{IMG_DIR}/{image_copied_name}.png")
    box_idx1 = df[df["copied_imageUid"] == image_copied_name].iloc[:,11:15].values
    box_name1 = df[df["copied_imageUid"] == image_copied_name].iloc[:,1].values
    img2 = cv2.imread(f"{IMG_DIR}/{image_based_name}.png")
    box_idx2 = df[df["based_imageUid"] == image_based_name].iloc[:,5:9].values
    box_name2 = df[df["based_imageUid"] == image_based_name].iloc[:,1].values
    
    for idx1, name1 in zip(box_idx1, box_name1):
        img1 = cv2.rectangle(img1, (int(idx1[0]), int(idx1[1])), (int(idx1[2]), int(idx1[3])), (0,0,255), 2)
        img1 = cv2.putText(img1, name1, (int(idx1[0]), int(idx1[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    for idx2, name2 in zip(box_idx2, box_name2):
        img2 = cv2.rectangle(img2, (int(idx2[0]), int(idx2[1])), (int(idx2[2]), int(idx2[3])), (0,0,255), 2)
        img2 = cv2.putText(img2, name2, (int(idx2[0]), int(idx2[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    
    img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
    img = cv2.hconcat([img1, img2])

    cv2.imwrite(f"{IMG_BOX_DIR}/{image_copied_name}_{image_based_name}.png", img)

for index, row in df.iterrows():
    try:   
        draw_bboxes_compare_two_sequences(row['based_imageUid'], row['copied_imageUid'])
    except Exception as e:
        print(e)

    

# %%
list_studies = list(df_copied["studyUid"].drop_duplicates())
for stu in list_studies:
    num_t1c = len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["BasedSeriesLabel"]=="T1C")]) + len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["CopiedSeriesLabel"]=="T1C")])
    num_t2 = len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["BasedSeriesLabel"]=="T2")]) + len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["CopiedSeriesLabel"]=="T2")])
    num_flair = len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["BasedSeriesLabel"]=="FLAIR")]) + len(df_copied[(df_copied["studyUid"]==stu) & (df_copied["CopiedSeriesLabel"]=="FLAIR")])
    print(f"StudiesUid {stu} has {num_t1c} T1C, {num_t2} T2, and {num_flair} FLAIR")


# %%
# Create 3-channels image from 3 slices, order by ["FLAIR", "T1C", "T2"]
# ORDER_SEQUENCES = {"FLAIR": 0, "T1C": 1, "T2": 2}
# list_order_imageUid = []
# for index, row in df_copied.iterrows():
#     order_imageUid = []
    
# %%


# %%


# %%

