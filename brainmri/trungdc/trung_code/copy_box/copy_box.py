#%%
# import library
import os
import cv2
# import pickle5 as pickle
import pickle
import pydicom
from tqdm import tqdm
import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut

#%%
class Error(Exception):
    pass
class MultipleSeriesError(Error):
    def __init__(self, message):
        self.message = message

#%%
# set global variable
root_images_PNG = "/home/single3/Documents/tintrung/test_im"
summary_dicom_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/summary_dicom.pkl"
summary_anot_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset.pkl"

#%%
df_dicom = pd.read_pickle(summary_dicom_path)
# df_dicom_extend: Add column "z-axis-ImagePositionPatient", notice that in df_dicom['ImagePositionPatient'] column have None values
z_axis = []
for item in tqdm(list(df_dicom['ImagePositionPatient'])):
    if item != None:
        z_axis.append(item[-1])
    else:
        z_axis.append(None)
df_dicom['z-axis-ImagePositionPatient'] = z_axis
df_dicom.head()

#%%

df_anot = pd.read_pickle(summary_anot_path)
df_anot = df_anot[(df_anot.label == "Mass/Nodule") | (df_anot.label == 'Cerebral edema')]
column_z = []
column_iop = []
for index, row in tqdm(df_anot.iterrows()):
    origin_filename = row["imageUid"] + ".dcm"
    match_df = df_dicom[df_dicom.DicomFileName == origin_filename]
    match_z = match_df["ImagePositionPatient"].values[0][-1]
    column_z.append(match_z)
    match_iop = match_df["ImageOrientationPatient"].values[0]
    column_iop.append(match_iop)
df_anot["z"] = column_z
df_anot["ImagePositionPatient"] = column_iop
df_anot.head()

#%%
def min_distance(given_point: float, list_points: list):
    """
    Find a point of list point that has minimum distance with given point
    """
    list_distances = [np.abs(given_point - pt) for pt in list_points]
    index_min = np.argmin(list_distances)
    # print(list_distances)
    target_point = float(list_points[index_min])
    # print(target_point-given_point)
    return [index_min, target_point]

# %%
list_studiesuid = list(df_anot['studyUid'].drop_duplicates())
print(f"Number of study: {len(list_studiesuid)}")

#%%
studyUid = []
seriesUid = []
imageUid = []
label = []
x1 = []
x2 = []
y1 = []
y2 = []
z = []

chosen_studies = list_studiesuid[0]
print(f"Chosen studyuid: {chosen_studies}")
try:
    # extract annotations corresponding to studyuid
    df_anot_extract = df_anot[df_anot['studyUid']==chosen_studies].reset_index()
    print(f"Number of slice in annotated series: {len(df_anot_extract)}")

    # check whether study has more than one annotated series uid
    # if that is the case, raise error
    annotated_series = list(df_anot_extract["seriesUid"].drop_duplicates().values)
    if len(annotated_series) != 1:
        raise MultipleSeriesError("This study has more than one annotated series")
    annotated_series = annotated_series[0]
    print(f"Annotated series: {annotated_series}")

    # extract dicom files corresponding to studyuis
    df_dcm_extract = df_dicom[df_dicom['StudyInstanceUID']==chosen_studies].reset_index()
    
    # extract list of seriesuids to be copied from given studyuid
    unprocess_series = list(df_dcm_extract[df_dcm_extract.StudyInstanceUID == chosen_studies]["SeriesInstanceUID"].drop_duplicates().values)
    unprocess_series.remove(annotated_series)   # remove annotated series
    print(f"Number of series to be copied: {len(unprocess_series)}")

    for cur_seriesuid in unprocess_series:
        unique_iop = df_dicom[df_dicom.SeriesInstanceUID == cur_seriesuid]["ImageOrientationPatient"].drop_duplicates().values
        print(cur_seriesuid)
        print(unique_iop[0])

    # print(len(df_anot_1_studies))
    lst_z_anot = list(df_anot_extract['z'])
    
    for idx, row in tqdm(df_dcm_1_studies.iterrows()):
        img_df_dcm = cv2.imread(os.path.join(root_images_PNG, row['DicomFileName'].replace('.dcm', '.png')))
        w1, h1, _ = img_df_dcm.shape
        base_z = row['z-axis-ImagePositionPatient']
        idx_min, choisen_z_in_lst = min_distance(base_z, lst_z_anot)
        chosen_row_anot = df_anot_1_studies.iloc[idx_min]
        if abs(base_z - choisen_z_in_lst) < row['PixelSpacing'][0]:
            studyUid.append(chosen_studies) ;'./>'
            seriesUid.append(row['SeriesInstanceUID'])
            imageUid.append(row['DicomFileName'].split('.dcm')[0])
            label.append(chosen_row_anot['label'])
            img_df_anot = cv2.imread(os.path.join(root_images_PNG, chosen_row_anot['imageUid']+'.png'))
            w2, h2, _ = img_df_anot.shape
            ratio = w1 / w2
            x1.append(int(chosen_row_anot['x1']*ratio))
            x2.append(int(chosen_row_anot['x2']*ratio))
            y1.append(int(chosen_row_anot['y1']*ratio))
            y2.append(int(chosen_row_anot['y2']*ratio))
            z.append(chosen_row_anot['z'])
except MultipleSeriesError as mse:
    print(f"Error study with more than one annotated series: {chosen_studies}")
    studies = list(df_anot_1_studies["seriesUid"].drop_duplicates().values).drop_duplicates()
    print(f"Number of annotated study: {len(studies)}")
    print(*studies, sep= "\n")



data_bbox_copy = {
    "studyUid": studyUid,
    "seriesUid": seriesUid,
    "imageUid": imageUid,
    "label": label,
    "x1": x1,
    "x2": x2,
    "y1": y1,
    "y2": y2,
    "z": z,
}
# print(data_bbox_copy)

df_bbox_copy = pd.DataFrame(data_bbox_copy, columns=["studyUid", "seriesUid", "imageUid", "label", "x1", "x2", "y1", "y2"])
df_bbox_copy.to_pickle("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy_trung.pkl")

  # %%
dm = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy_trung.pkl")
print(len(dm))
dm
# %%
