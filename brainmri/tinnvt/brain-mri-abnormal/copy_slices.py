#%%%
# import library
import os
import cv2
# import pickle5 as pickle
import pickle
import pydicom
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
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
root_images_PNG = "/home/single3/tintrung/brain-mri-tumor-images-PNG"
summary_dicom_path = "/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/summary_dicom.pkl"
summary_anot_path = "/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset.pkl"

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
df["ImageOrientationPatient"] = column_iop
df

#%%%%%%%%%%%
df_dicom_extend

# #%%%
# # Experiment with 1 studiesUID
# ## Choice 1 studiesUID and list all seriesUID
# based_series_in_one_choice_studies = list(df[df['studyUid']=='1.2.840.113619.6.388.106738880751051415278716030635969460033']['seriesUid'].drop_duplicates())
# all_series_in_one_choice_studies = list(df_dicom_extend[df_dicom_extend['StudyInstanceUID']=='1.2.840.113619.6.388.106738880751051415278716030635969460033']['SeriesInstanceUID'].drop_duplicates())
# print('based_series_in_one_choice_studies:\n', based_series_in_one_choice_studies)
# print('all_series_in_one_choice_studies:\n', all_series_in_one_choice_studies)

# iop_based = list(df_dicom_extend[df_dicom_extend['SeriesInstanceUID']==based_series_in_one_choice_studies[0]]['ImageOrientationPatient'])[0]
# iop_based = [[iop_based[i], iop_based[i+3]] for i in range(3)]
# iop_based = np.array(iop_based).astype(np.float32)

# iop_need_copied = list(df_dicom_extend[df_dicom_extend['SeriesInstanceUID']==all_series_in_one_choice_studies[0]]['ImageOrientationPatient'])[0]
# iop_need_copied = [[iop_need_copied[i], iop_need_copied[i+3]] for i in range(3)]
# iop_need_copied = np.array(iop_need_copied).astype(np.float32)

# matrix_affine = cv2.getAffineTransform(iop_need_copied, iop_based)

# print('iop_based:\n', iop_based)
# print('iop_need_copied:\n', iop_need_copied)
# print('Affine Transformation:\n', matrix_affine)
# # dst = cv2.warpAffine(img, M, (cols, rows))

#%%
def create_vector_from_iop(iop):
    """
    IOP into vector in 3D-dimensions
    """
    vector = [iop[i]-iop[i+3] for i in range(3)]
    return vector


def length_vector(vec3d):
    """
    Return length of a vector in 3d dimensions
    """
    return np.sqrt(vec3d[0]**2 + vec3d[1]**2 + vec3d[2]**2)


def angle_of_two_3dvector(vec1, vec2):
    """
    Return angle of two vector in 3d dimensions
    """
    if (vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]) > 0:
        angle = np.arccos((vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]) / (length_vector(vec1) * length_vector(vec2)))
    else:
        angle = np.pi - np.arccos((vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]) / (length_vector(vec1) * length_vector(vec2)))
    return angle


def process_rotation_points(src_2dpoint, iop_base, iop_copied):
    """
    
    """
    vec_base = create_vector_from_iop(iop_base)
    vec_copied = create_vector_from_iop(iop_copied)
    rotation_angle_radians = angle_of_two_3dvector(vec_copied, vec_base)

    rotation_axis = np.array([1, 1, 1])
    rotation_vector = rotation_angle_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    src_extend_3dpoint = np.array([src_2dpoint[0], src_2dpoint[1], 0])
    rotated_vec = rotation.apply(src_extend_3dpoint)
    return rotated_vec[0], rotated_vec[1]

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
list_x1 = []
list_x2 = []
list_y1 = []
list_y2 = []
list_z = []

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
                    img_df_anot = cv2.imread(os.path.join(root_images_PNG, chosen_row_anot['imageUid']+'.png'))
                    w2, h2, _ = img_df_anot.shape
                    
                    # Process rotation
                    iop_base = row['ImageOrientationPatient']
                    iop_copied = chosen_row_anot['ImageOrientationPatient']
                    x1, x2 = process_rotation_points([x1,x2], iop_base, iop_copied)
                    y1, y2 = process_rotation_points([y1,y2], iop_base, iop_copied)

                    # Process ratio of source img and target img
                    ratio = w1 / w2
                    x1 = chosen_row_anot['x1']*ratio
                    x2 = chosen_row_anot['x2']*ratio
                    y1 = chosen_row_anot['y1']*ratio
                    y2 = chosen_row_anot['y2']*ratio

                    # print(x1)
                    # print(x2)
                    # print(y1)
                    # print(y2)

                    if x1==None:
                        list_x1.append(None)
                    else:
                        list_x1.append(int(x1))

                    if x2==None:
                        list_x2.append(None)
                    else:
                        list_x2.append(int(x2))

                    if y1==None:
                        list_y1.append(None)
                    else:
                        list_y1.append(int(y1))

                    if y2==None:
                        list_y2.append(None)
                    else:
                        list_y2.append(int(y2))

                    if chosen_row_anot['z']==None:
                        list_z.append(None)
                    else:
                        list_z.append(chosen_row_anot['z'])

                    # list_x1.append(int(x1))
                    # list_x2.append(int(x2))
                    # list_y1.append(int(y1))
                    # list_y2.append(int(y2))
                    # list_z.append(chosen_row_anot['z'])

                    studyUid.append(chosen_studies)
                    seriesUid.append(row['SeriesInstanceUID'])
                    imageUid.append(row['DicomFileName'].split('.dcm')[0])
                    label.append(chosen_row_anot['label'])
                
    except:
        print(f"Error study with more than one annotated series: {chosen_studies}")
        studies = list(df_anot_1_studies["seriesUid"].drop_duplicates().values)
        print(f"Number of annotated study: {len(studies)}")
        print(*studies, sep= "\n")

print(len(studyUid))
print(len(seriesUid))
print(len(imageUid))
print(len(label))
print(len(list_x1))
print(len(list_x2))
print(len(list_y1))
print(len(list_y2))
print(len(list_z))



data_bbox_copy = {
    "studyUid": studyUid,
    "seriesUid": seriesUid,
    "imageUid": imageUid,
    "label": label,
    "x1": list_x1,
    "x2": list_x2,
    "y1": list_y1,
    "y2": list_y2,
    "z": list_z,
}
# print(data_bbox_copy)

df_bbox_copy = pd.DataFrame(data_bbox_copy, columns=["studyUid", "seriesUid", "imageUid", "label", "x1", "x2", "y1", "y2"])
df_bbox_copy.to_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl")

# %%
dm = pd.read_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl")
print(len(dm))
dm
# %%
