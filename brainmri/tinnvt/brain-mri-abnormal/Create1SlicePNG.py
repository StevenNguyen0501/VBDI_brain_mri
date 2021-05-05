import os
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_dicom(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


def convertDCMFolder2png(pathDCMFolder: str, pathPNGFolder: str):
    """
    Convert all dicom files to png format images
    """
    for dcmName in tqdm(os.listdir(pathDCMFolder)):
        try:
            dcmPth = os.path.join(pathDCMFolder, dcmName)
            img = read_dicom(dcmPth)
            plt.imsave(os.path.join(pathPNGFolder, dcmName.replace('.dcm', '.png')), img)
        except Exception as error:
            print(f'File dicom {dcmName} has error with {error}')


pathDCMs = '/home/single3/Documents/tintrung/brain-mri-tumor-dicom-masked'
pathPNGs = '/home/single3/Documents/tintrung/brain-mri-tumor-PNG'

for subdir in tqdm(os.listdir(pathDCMs)):
    pathSubDir = os.path.join(pathDCMs, subdir)
    convertDCMFolder2png(pathSubDir, pathPNGs)