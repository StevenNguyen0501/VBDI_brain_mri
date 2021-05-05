import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

OLD_IMG_DIR = "/media/datnt/data/medical-image-data/brain-mri-images"
NEW_IMG_DIR = "/media/datnt/data/medical-image-data/brain-mri-images2"

for set_ in ["train", "valid", "holdout"]:
    df = pd.read_csv(f"csv/brain-mri-abnormalness-{set_}-v2.csv")
    for s in tqdm(df["studyUid"].unique()):
        sub_df = df[df["studyUid"] == s].sort_values(by=["loc"])
        for i in range(0, len(sub_df)):
            if i == 0:
                img_names = [sub_df.iloc[i, 1]] + list(sub_df.iloc[i:i+2, 1].values)
            elif i == len(sub_df)-1:
                img_names = list(sub_df.iloc[i-1:i+1, 1].values) + [sub_df.iloc[i, 1]]
            else:
                img_names = sub_df.iloc[i-1:i+2, 1].values
            # img0 = cv2.cvtColor(cv2.imread(os.path.join(OLD_IMG_DIR, img_names[0] + ".jpg")),cv2.COLOR_RGB2GRAY)
            # img1 = cv2.cvtColor(cv2.imread(os.path.join(OLD_IMG_DIR, img_names[1] + ".jpg")),cv2.COLOR_RGB2GRAY)
            # img2 = cv2.cvtColor(cv2.imread(os.path.join(OLD_IMG_DIR, img_names[2] + ".jpg")),cv2.COLOR_RGB2GRAY)
            
            # img = np.zeros(shape=(512,512,3), dtype=np.uint8)
            # try:
            #     img[:,:,0] = img0
            #     img[:,:,1] = img1
            #     img[:,:,2] = img2
            # except:
            #     img0 = cv2.resize(img0, (512,512))
            #     img1 = cv2.resize(img1, (512,512))
            #     img2 = cv2.resize(img2, (512,512))
            #     img[:,:,0] = img0
            #     img[:,:,1] = img1
            #     img[:,:,2] = img2
            # name = sub_df.iloc[i]["imageUid"]
            # cv2.imwrite(f"{NEW_IMG_DIR}/{name}.jpg", img)
            print(img_names)
            
        
    