import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut


def dicom2array(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.filereader.dcmread(path, stop_before_pixels = True)
    # VOI LUT (if available by DICOM device) is used to
    # transform raw DICOM data to "human-friendly" view
    # if voi_lut:
    #     data = apply_voi_lut(dicom.pixel_array, dicom)
    # else:
    #     data = dicom.pixel_array
    # # depending on this value, X-ray may look inverted - fix that:
    # if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
    #     data = np.amax(data) - data
    # data = data - np.min(data)
    # data = data / np.max(data)
    # data = (data * 255).astype(np.uint8)
    return dicom #, data
        
    
def plot_img(img, size=(7, 7), is_rgb=True, title="", cmap='gray'):
    plt.figure(figsize=size)
    plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def plot_imgs(imgs, cols=4, size=7, is_rgb=True, title="", cmap='gray', img_size=(500,500)):
    rows = len(imgs)//cols + 1
    fig = plt.figure(figsize=(cols*size, rows*size))
    for i, img in enumerate(imgs):
        if img_size is not None:
            img = cv2.resize(img, img_size)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img, cmap=cmap)
    plt.suptitle(title)
    plt.show()


def get_infor_dicom(dicom):
    row = []
    
    if 'Study Instance UID' in str(dicom):
        row.append(dicom.StudyInstanceUID)
    else: 
        row.append(None)
    
    if 'Series Instance UID' in str(dicom):
        row.append(dicom.SeriesInstanceUID)
    else: 
        row.append(None)

    if 'Series Description' in str(dicom):
        row.append(dicom.SeriesDescription)
    else: 
        row.append(None)

    if 'Rows' in str(dicom):
        row.append(dicom.Rows)
    else: 
        row.append(None)

    if 'Columns' in str(dicom):
        row.append(dicom.Columns)
    else: 
        row.append(None)

    if 'Slice Thickness' in str(dicom):
        row.append(dicom.SliceThickness)
    else: 
        row.append(None)

    if 'Pixel Spacing' in str(dicom):
        row.append(dicom.PixelSpacing)
    else: 
        row.append(None)

    if 'Spacing Between Slices' in str(dicom):
        row.append(dicom.SpacingBetweenSlices)
    else: 
        row.append(None)

    if 'Slice Location' in str(dicom):
        row.append(dicom.SliceLocation)
    else: 
        row.append(None)

    if 'Smallest Image Pixel Value' in str(dicom):
        row.append(dicom.SmallestImagePixelValue)
    else: 
        row.append(None)

    if 'Largest Image Pixel Value' in str(dicom):
        row.append(dicom.LargestImagePixelValue)
    else: 
        row.append(None)

    if 'Window Center' in str(dicom):
        row.append(dicom.WindowCenter)
    else: 
        row.append(None)

    if 'Window Width' in str(dicom):
        row.append(dicom.WindowWidth) 
    else: 
        row.append(None)

    if 'Image Position (Patient)' in str(dicom):
        row.append(dicom.ImagePositionPatient)
    else: 
        row.append(None)

    if 'Image Orientation (Patient)' in str(dicom):
        row.append(dicom.ImageOrientationPatient)
    else: 
        row.append(None)

    if 'Repetition Time' in str(dicom):
        row.append(dicom.RepetitionTime)
    else: 
        row.append(None)

    if 'Echo Time' in str(dicom):
        row.append(dicom.EchoTime)
    else: 
        row.append(None)

    if 'Patient Position' in str(dicom):
        row.append(dicom.PatientPosition)
    else: 
        row.append(None)

    return row


dcm_root_path = "/home/single3/Documents/tintrung/yolov5_2classes/test/1.2.840.113619.6.408.217046082646534226129609417598798276891/1.2.840.113619.6.408.217046082646534226129609417598798276891"
dcm_paths = []

for subdir_name in os.listdir(dcm_root_path):
    subdir_path = os.path.join(dcm_root_path, subdir_name)
    for file_name in os.listdir(subdir_path):
        file_path = os.path.join(subdir_path, file_name)
        dcm_paths.append(file_path)
print("The total number of dicom files:", len(dcm_paths))

data_dcm = []
kye_cols_dcm = ['StudyInstanceUID', 
                'SeriesInstanceUID', 
                'SeriesDescription', 
                'Rows', 
                'Columns', 
                'SliceThickness', 
                'PixelSpacing', 
                'SpacingBetweenSlices', 
                'SliceLocation', 
                'SmallestImagePixelValue', 
                'LargestImagePixelValue', 
                'WindowCenter', 
                'WindowWidth', 
                'ImagePositionPatient', 
                'ImageOrientationPatient', 
                'RepetitionTime', 
                'EchoTime', 
                'PatientPosition', 
                'NameFileDCM']

for dcm_path in dcm_paths:  
    dicom = dicom2array(dcm_path)
    print(dicom)
    # Name of xml file
#     name_dicom = dcm_path.split('/')[-1]
#     print(name_dicom)
#     row = get_infor_dicom(dicom)
#     row.append(name_dicom)
#     data_dcm.append(row)
#     print(len(data_dcm))
    
    
# data_dcm = np.array(data_dcm)
# df_dcm = pd.DataFrame(data_dcm, columns=kye_cols_dcm)
# print(df_dcm)
# df_dcm.to_excel('/home/single3/Documents/tintrung/yolov5_2classes/test/1.2.840.113619.6.408.217046082646534226129609417598798276891/summary_dicom_test.xlsx')
# df_dcm.to_pickle('/home/single3/Documents/tintrung/yolov5_2classes/test/1.2.840.113619.6.408.217046082646534226129609417598798276891/summary_dicom_test.pkl')