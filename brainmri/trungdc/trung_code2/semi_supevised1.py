
#%%
import pandas as pd 
import tqdm
from tqdm import tqdm
from math import acos
from numpy import add, degrees
import numpy as np

import os
import matplotlib.pyplot as plt

from datetime import datetime

#%%
now = datetime.now()
dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
os.chdir("/home/single2/tintrung/VBDI_brain_mri/brainmri/trungdc/")
bbox = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/brainmri/trungdc/pkl_ver1/bbox_dat_extend_sequence.pkl")
bbox.head()

#%%
dataset = pd.read_pickle("pkl_ver1/summary_dicom_extend.pkl")

count = 0
images = pd.DataFrame()
for column in  [ "id", "file_name", "height", "width", "date_captured","category_id","bbox","sequence", "studyUid", "seriesUid"]:
    images[column] = []
for index, row in tqdm(bbox.copy().iterrows()):
    extract_dataset = dataset[dataset.DicomFileName == (row["imageUid"] + ".dcm")]
    rows = list(extract_dataset["Rows"].values)[0]
    columns = list(extract_dataset["Columns"].values)[0]
    # print((rows,columns))

    label = row["label"]
    if label == "Mass/Nodule":
        label = 1
    else:
        label = 0

    cur_sequence = row["SeriesLabelMerge"]

    topleftx = row["x1"]
    toplefty = row["x2"]
    height = abs(row["x2"] - row["y2"])
    width = abs(row["x1"] - row["y1"])
    bbox_coco_format = [topleftx, toplefty, width, height]

    id = row["imageUid"]
    id = int(id.replace(".", ""))

    addrow = [id, row["imageUid"] + ".png", columns, rows
        , dt_string,label,bbox_coco_format, cur_sequence, row["studyUid"], row["seriesUid"]]
    images.loc[count] = addrow 
    count += 1
images.head()


#%%
images.to_pickle("semisupervised/coco_bbox.pkl")
# %%
