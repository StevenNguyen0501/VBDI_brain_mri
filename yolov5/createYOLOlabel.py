#%%
from tqdm import tqdm
import os
import cv2
import pandas as pd
import numpy as np
import shutil
import json



#%%
PKL_FILE = "/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_5k_oldnew.pkl"
IMG_DIR = "/home/single2/tintrung/images/"
TXT_DIR = "/home/single2/tintrung/labels"

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', None)

df=  pd.read_pickle(PKL_FILE)
df.head()
# len(df)

#%%
CLASS_INDEX = {
 'Mass/Nodule': 0,
 'Cerebral edema': 1,
}

df_all_classes = pd.read_pickle(PKL_FILE)
df_2_classes = df_all_classes[(df_all_classes['label']=='Mass/Nodule') | (df_all_classes['label']=='Cerebral edema')]
listImages = list(df_2_classes['imageUid'].drop_duplicates())

for image in tqdm(listImages):
    sub_df = df_2_classes[df_2_classes["imageUid"] == image].sort_values(by='label')  
    sub_df = sub_df.reset_index() 
    # (x1, x2), (y1, y2) are 2 points
    print(os.path.exists(os.path.join(IMG_DIR, image+".png")))
    img = cv2.imread(os.path.join(IMG_DIR, image+".png"))
    h, w, _ = img.shape

    className = []
    x_center = []
    y_center = []
    width = []
    height = []
    for j in range(sub_df.shape[0]):
        className.append(int(CLASS_INDEX[sub_df['label'][j]]))
        x_center.append(float(((sub_df['x1'][j] + sub_df['y1'][j]) / 2) / w))
        y_center.append(float(((sub_df['x2'][j] + sub_df['y2'][j]) / 2) / h))
        width.append(float((sub_df['y1'][j] - sub_df['x1'][j]) / w))
        height.append(float((sub_df['y2'][j] - sub_df['x2'][j]) / h))
    df_result = pd.DataFrame(data={'className': className, 
                                   'x_center': x_center, 
                                   'y_center': y_center, 
                                   'width': width, 
                                   'height': height})
    path_txt_file = os.path.join(TXT_DIR, image+".txt")
    np.savetxt(path_txt_file, df_result.values, delimiter="\t")
# %%
