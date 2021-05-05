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
class SeriesError(Error):
    def __init__(self, message):
        self.message = message

#%%
# set global variable
root_images_PNG = "/home/single3/Documents/tintrung/brain-mri-tumor-images-PNG"
summary_dicom_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/summary_dicom.pkl"
summary_anot_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset.pkl"

#%%
df_dicom = pd.read_pickle(summary_dicom_path)
# df_dicom_extend: Add column "z-axis-ImagePositionPatient", notice that in df_dicom['ImagePositionPatient'] column have None values
df_dicom_extend = df_dicom
z_axis = []
for item in list(df_dicom['ImagePositionPatient']):
    if item != None:
        z_axis.append(item[-1])
    else:
        z_axis.append(None)
df_dicom_extend['z-axis-ImagePositionPatient'] = z_axis

df = pd.read_pickle(summary_anot_path)
df = df[(df.label == "Mass/Nodule") | (df.label == 'Cerebral edema')]
column_z = []
column_iop = []
for index, row in tqdm(df.iterrows()):
    origin_filename = row["imageUid"] + ".dcm"
    assert origin_filename in list(df_dicom["DicomFileName"].drop_duplicates())
    match_df = df_dicom[df_dicom.DicomFileName == origin_filename]
    assert len(match_df) == 1
    match_z = match_df["ImagePositionPatient"].values[0][-1]
    column_z.append(match_z)
    match_iop = match_df["ImageOrientationPatient"].values[0]
    column_iop.append(match_iop)
df["z"] = column_z
df["ImagePositionPatient"] = column_iop
df

#%%%%%%%%%%%
df_dicom_extend

#%%
def min_distance(given_point: float, list_points: list):
    """
    Find a point of list points that has minimum distance with given point
    """
    list_distances = [np.abs(given_point - pt) for pt in list_points]
    index_min = np.argmin(list_distances)
    # print(list_distances)
    target_point = float(list_points[index_min])
    # print(target_point-given_point)
    return [index_min, target_point]

# %%
list_dcm_remove = list(df['imageUid'].drop_duplicates())
list_studiesuid = list(df['studyUid'].drop_duplicates())

studyUid = []
seriesUid = []
imageUid = []
label = []
x1 = []
x2 = []
y1 = []
y2 = []
z = []

for chosen_studies in list_studiesuid:
    try:
        df_anot_1_studies = df[df['studyUid']==chosen_studies].reset_index()
        if len(list(df_anot_1_studies["seriesUid"].drop_duplicates().values)) != 1:
            raise Exception("Error")

        df_dcm_1_studies = df_dicom_extend[df_dicom_extend['StudyInstanceUID']==chosen_studies].reset_index()
        
        # print(len(df_anot_1_studies))
        lst_z_anot = list(df_anot_1_studies['z'])
        
        for idx, row in tqdm(df_dcm_1_studies.iterrows()):
            if not row['DicomFileName'].split('.dcm')[0] in list_dcm_remove:
                img_df_dcm = cv2.imread(os.path.join(root_images_PNG, row['DicomFileName'].replace('.dcm', '.png')))
                w1, h1, _ = img_df_dcm.shape
                base_z = row['z-axis-ImagePositionPatient']
                idx_min, choisen_z_in_lst = min_distance(base_z, lst_z_anot)
                chosen_row_anot = df_anot_1_studies.iloc[idx_min]
                if abs(base_z - choisen_z_in_lst) < row['PixelSpacing'][0]:
                    studyUid.append(chosen_studies)
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
    except:
        print(f"Error study with more than one annotated series: {chosen_studies}")
        studies = list(df_anot_1_studies["seriesUid"].drop_duplicates().values)
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
df_bbox_copy.to_pickle("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl")

# %%
dm = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl")
print(len(dm))
dm
# %%
