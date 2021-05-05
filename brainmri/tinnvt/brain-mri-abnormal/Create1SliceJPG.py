import os
import pydicom
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd

DCM_DIR = "/media/datnt/data/medical-image-data/brain-mri-dicom"
JPG_DIR = "/media/datnt/data/medical-image-data/brain-mri-images"

for set_ in ["train", "valid", "holdout"]:
    df = pd.read_csv(f"csv/brain-mri-abnormalness-{set_}-v2.csv")
    snames = df["studyUid"].value
    fnames = df["imageUid"].value
    for f, s in tqdm(zip(fnames, snames)):
        try:
            data = pydicom.read_file(os.path.join(DCM_DIR, s, "DICOM", f + ".dcm"), force=True)
        except:
            data = pydicom.read_file(os.path.join(DCM_DIR, s, "DICOM", f + ".dicom"), force=True)
        img = data.pixel_array
        img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, data)
        if data.PhotometricInterpretation == "MONOCHROME1":
            img = np.amax(img) - img
        img = img - np.min(img)
        img = img / np.max(img)
        img = (img * 255).astype(np.uint8)
        iname = f.split("/")[-1].replace("dcm", "jpg").replace("dicom", "jpg")
        cv2.imwrite(f"{JPG_DIR}/{iname}", img)
