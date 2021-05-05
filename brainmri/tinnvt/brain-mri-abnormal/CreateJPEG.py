import pandas as pd
from tqdm import tqdm
import os
import pydicom
import numpy as np
import cv2

CSV_FILE = "brain-mri-dataset-batch2.csv"
DICOM_DIR = "/media/datnt/data/medical-image-data/brain-mri-dicom"
DEST_DIR = "/media/datnt/data/medical-image-data/brain-mri-images"

df = pd.read_csv(CSV_FILE).drop_duplicates(subset=["imageUid"])
for i in tqdm(range(len(df))):
    study = df.iloc[i, 0]
    instance = df.iloc[i,2]
    try:
        data = pydicom.read_file(os.path.join(DICOM_DIR, study, "DICOM", instance + ".dcm"), force=True)
    except:
        data = pydicom.read_file(os.path.join(DICOM_DIR, study, "DICOM", instance + ".dicom"), force=True)
    img = data.pixel_array
    img = pydicom.pixel_data_handlers.util.apply_voi_lut(img, data)
    if data.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img
    img = img - np.min(img)
    img = img / np.max(img)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DEST_DIR, instance + ".jpg"), img)