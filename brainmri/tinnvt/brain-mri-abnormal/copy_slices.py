#%%%
# import library
import os
import cv2
import math
# import pickle5 as pickle
import pickle
from numpy.core.numeric import NaN
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
column_ipp = []
column_pixelSpacing = []
for index, row in tqdm(df.iterrows()):
    origin_filename = row["imageUid"] + ".dcm"
    assert origin_filename in list(df_dicom["DicomFileName"].drop_duplicates())
    match_df = df_dicom[df_dicom.DicomFileName == origin_filename]
    assert len(match_df) == 1
    match_ipp = match_df["ImagePositionPatient"].values[0]
    column_ipp.append(match_ipp)
    match_z = match_df["ImagePositionPatient"].values[0][-1]
    column_z.append(match_z)
    match_iop = match_df["ImageOrientationPatient"].values[0]
    column_iop.append(match_iop)
    match_pixelSpace = match_df["PixelSpacing"].values[0]
    column_pixelSpacing.append(match_pixelSpace)
df["z"] = column_z
df["ImagePositionPatient"] = column_ipp
df["ImageOrientationPatient"] = column_iop
df["PixelSpacing"] = column_pixelSpacing
df.head()

#%%%%%%%%%%%
pd.set_option('display.max_colwidth', None)
df_dicom_extend.tail()

#%%
# #%%%
# l = []
# list_copy_fail_slice = ['1.2.840.113619.2.80.2802896632.23344.1591580541.10', '1.2.840.113619.2.80.2802896632.23344.1591580543.38', '1.2.840.113619.2.311.224079951486824588700645495241844903444', '1.2.840.113619.2.388.57473.14165493.18300.1597361334.760', '1.2.840.113619.2.410.15512023.5814788.14684.1586132594.104']
# list_copy_true_slice = ['1.2.840.113619.2.388.57473.14165493.11772.1596151049.764', '1.2.840.113619.2.388.57473.14165493.11772.1596151024.278', '1.2.840.113619.2.311.328285041751902862821898989135346683846', '1.2.840.113619.2.311.284244468119992884979176337237919287967', '1.2.840.113619.2.311.248899921507219449064122357390320698635', '1.2.840.113619.2.311.179293298660890413807912849283784678654', '1.2.840.113619.2.311.163956421163993427281360692960682270033', '1.2.840.113619.2.311.149787184296461495080357141051443890030']
# for i in list_copy_fail_slice:
#     i = i+'.dcm'
#     l.append(df_dicom_extend[df_dicom_extend['DicomFileName']==i]['StudyInstanceUID'])
# l


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
    vector = [iop[i]-iop[i-3] for i in range(2, -1, -1)]
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


def process_rotation_ipp(src_2dpoint, ipp_base, ipp_copied, pixelSpacingBase, pixelSpacingCopied, ratioScaleCopied2Base):
    """
    
    """
    # Convert real-world coordinate to image coordinate, then calculate the coordinate of points after scaling
    ipp_base = [int(item//pixelSpacingBase*ratioScaleCopied2Base) for item in ipp_base]
    # print(pixelSpacingCopied)
    # print(ratioScaleCopied2Base)
    ipp_copied = [int(item//pixelSpacingCopied) for item in ipp_copied]
    # Translate points basing on Image Position Patient
    trans_vec = [int(ipp_copied[i]-ipp_base[i]) for i in range(len(ipp_base))]
    src_2dpoint = [int(src_2dpoint[i]+trans_vec[i]) for i in range(len(src_2dpoint))]
    return src_2dpoint


def process_rotation_iop(src_2dpoint, iop_base, iop_copied):
    """
    
    """
    vec_base = create_vector_from_iop(iop_base)
    vec_copied = create_vector_from_iop(iop_copied)
    rotation_angle_radians = angle_of_two_3dvector(vec_base, vec_copied)

    rotation_axis = np.array([1, 1, 1])
    rotation_vector = rotation_angle_radians * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    src_extend_3dpoint = np.array([src_2dpoint[0], src_2dpoint[1], 0])
    rotated_vec = rotation.apply(src_extend_3dpoint)
    return rotated_vec[0], rotated_vec[1]


def process_copy_slices(point2d,
                        ipp_base, 
                        ipp_copied, 
                        pixelSpacingBase, 
                        pixelSpacingCopied, 
                        ratioScaleCopied2Base, 
                        iop_base, 
                        iop_copied):
    """
    
    """
    # point2d = process_rotation_ipp(point2d, ipp_base, ipp_copied, pixelSpacingBase, pixelSpacingCopied, ratioScaleCopied2Base)
    point2d = process_rotation_iop(point2d, iop_base, iop_copied)
    return point2d


# def min_distance(given_point: float, list_points: list):
#     """
#     Find a point of list points that has minimum distance with given point
#     """
#     list_distances = [np.abs(given_point - pt) for pt in list_points]
#     index_min = np.argmin(list_distances)
#     # print(list_distances)
#     target_point = float(list_points[index_min])
#     # print(target_point-given_point)
#     return [index_min, target_point]


def min_distance(given_point: float, list_points: list, threshold = 1):
    """
    Find a point of list points that has minimum distance with given point
    Arguments:
    @given_points: point to be matched
    @list_points: sorted list of point
    @threshold: maximum error between two x values
        default is 1 mm

    Return:
    If z is matched:
    @index: index of match slice in annotation dataframe
    @target_points: z value of matching slices
    @error: differences between given_point and matched_point
    """
    if not list_points:
        return None
    else:
        if (given_point >= min(list_points)) & (given_point <= max(list_points)):
            list_distances = [np.abs(given_point - pt) for pt in list_points]
            index = np.argmin(list_distances)
            target_point = float(list_points[index])
            error = abs(target_point - given_point)
            if error <= threshold:
                return [index, target_point, error]
            else:
                return None
        elif abs(given_point - min(list_points)) <= threshold:
            index = 0
            target_point = float(list_points[index])
            error = abs(target_point - given_point)
            return [index, target_point, error]
        elif abs(given_point - max(list_points)) <= threshold:
            index = len(list_points)-1
            target_point = float(list_points[-1])
            error = abs(target_point - given_point)
            return [index, target_point, error]
        else:
            return None



# %%
list_dcm_remove = list(df['imageUid'].drop_duplicates())
list_studiesuid = list(df['studyUid'].drop_duplicates())

list_studyUid = []
list_label = []
list_z = []

list_based_seriesUid = []
list_based_imageUid = []
list_based_x1 = []
list_based_x2 = []
list_based_y1 = []
list_based_y2 = []

list_copied_seriesUid = []
list_copied_imageUid = []
list_copied_x1 = []
list_copied_x2 = []
list_copied_y1 = []
list_copied_y2 = []


for chosen_studies in list_studiesuid:
    try:
        df_anot_1_studies = df[df['studyUid']==chosen_studies].reset_index()
        if len(list(df_anot_1_studies["seriesUid"].drop_duplicates().values)) != 1:
            raise Exception("Error")

        df_dcm_1_studies = df_dicom_extend[df_dicom_extend['StudyInstanceUID']==chosen_studies].reset_index()
        
        # print(len(df_anot_1_studies))
        lst_z_anot = list(df_anot_1_studies['z'])
        
        for idx, row in tqdm(df_dcm_1_studies.iterrows()):
            
            chosen_seri = row['SeriesInstanceUID']
            
            if not row['DicomFileName'].split('.dcm')[0] in list_dcm_remove:
                img_df_dcm = cv2.imread(os.path.join(root_images_PNG, row['DicomFileName'].replace('.dcm', '.png')))
                w1, h1, _ = img_df_dcm.shape
                z_ipp = row['z-axis-ImagePositionPatient']
                dis = min_distance(z_ipp, lst_z_anot, threshold = 2)
                
                if dis != None:
                    idx_min, choisen_z_in_lst, error = dis[0], dis[1], dis[2]
                    chosen_row_anot = df_anot_1_studies.iloc[idx_min]
                    
                    if abs(z_ipp - choisen_z_in_lst) < row['PixelSpacing'][0]:                        
                        list_studyUid.append(chosen_studies)
                        list_copied_seriesUid.append(chosen_seri)
                        list_copied_imageUid.append(row['DicomFileName'].split('.dcm')[0])
                        list_label.append(chosen_row_anot['label'])

                        list_based_seriesUid.append(chosen_row_anot['seriesUid'])
                        list_based_imageUid.append(chosen_row_anot['imageUid'])
                        list_based_x1.append(chosen_row_anot['x1'])
                        list_based_x2.append(chosen_row_anot['x2'])
                        list_based_y1.append(chosen_row_anot['y1'])
                        list_based_y2.append(chosen_row_anot['y2'])
                                            
                        img_df_anot = cv2.imread(os.path.join(root_images_PNG, chosen_row_anot['imageUid']+'.png'))
                        w2, h2, _ = img_df_anot.shape
                        ratio = w1 / w2
                        x1 = chosen_row_anot['x1']*ratio
                        x2 = chosen_row_anot['x2']*ratio
                        y1 = chosen_row_anot['y1']*ratio
                        y2 = chosen_row_anot['y2']*ratio
                        x1, x2 = process_copy_slices([x1,x2],
                                                    ipp_base=chosen_row_anot["ImagePositionPatient"], 
                                                    ipp_copied=row["ImagePositionPatient"], 
                                                    pixelSpacingBase=chosen_row_anot["PixelSpacing"][0], 
                                                    pixelSpacingCopied=row["PixelSpacing"][0], 
                                                    ratioScaleCopied2Base=ratio, 
                                                    iop_base=chosen_row_anot["ImageOrientationPatient"], 
                                                    iop_copied=row["ImageOrientationPatient"])
                        y1, y2 = process_copy_slices([y1,y2],
                                                    ipp_base=chosen_row_anot["ImagePositionPatient"], 
                                                    ipp_copied=row["ImagePositionPatient"], 
                                                    pixelSpacingBase=chosen_row_anot["PixelSpacing"][0], 
                                                    pixelSpacingCopied=row["PixelSpacing"][0], 
                                                    ratioScaleCopied2Base=ratio, 
                                                    iop_base=chosen_row_anot["ImageOrientationPatient"], 
                                                    iop_copied=row["ImageOrientationPatient"])

                        if x1==None or math.isnan(x1):
                            list_copied_x1.append(None)
                        else:
                            list_copied_x1.append(int(x1))

                        if x2==None or math.isnan(x2):
                            list_copied_x2.append(None)
                        else:
                            list_copied_x2.append(int(x2))

                        if y1==None or math.isnan(y1):
                            list_copied_y1.append(None)
                        else:
                            list_copied_y1.append(int(y1))

                        if y2==None or math.isnan(y2):
                            list_copied_y2.append(None)
                        else:
                            list_copied_y2.append(int(y2))

                        if chosen_row_anot['z']==None:
                            list_z.append(None)
                        else:
                            list_z.append(chosen_row_anot['z'])
                
        # break
    except:
        print(f"Error study with more than one annotated series: {chosen_studies}")
        studies = list(df_anot_1_studies["seriesUid"].drop_duplicates().values)
        print(f"Number of annotated study: {len(studies)}")
        print(*studies, sep= "\n")


print(len(list_studyUid))
print(len(list_label))
print(len(list_z))
print(len(list_based_seriesUid))
print(len(list_based_imageUid))
print(len(list_based_x1))
print(len(list_based_x2))
print(len(list_based_y1))
print(len(list_based_y2))
print(len(list_copied_seriesUid))
print(len(list_copied_imageUid))
print(len(list_copied_x1))
print(len(list_copied_x2))
print(len(list_copied_y1))
print(len(list_copied_y2))


data_bbox_copy = {
    "studyUid": list_studyUid,
    "label": list_label,
    "z": list_z,
    "based_seriesUid": list_based_seriesUid,
    "based_imageUid": list_based_imageUid,
    "based_x1": list_based_x1,
    "based_x2": list_based_x2,
    "based_y1": list_based_y1,
    "based_y2": list_based_y2,
    "copied_seriesUid": list_copied_seriesUid,
    "copied_imageUid": list_copied_imageUid,
    "copied_x1": list_copied_x1,
    "copied_x2": list_copied_x2,
    "copied_y1": list_copied_y1,
    "copied_y2": list_copied_y2,  
}

df_bbox_copy = pd.DataFrame(data_bbox_copy, columns=["studyUid", "label", "z", "based_seriesUid", "based_imageUid", "based_x1", "based_x2", "based_y1", "based_y2", "copied_seriesUid", "copied_imageUid", "copied_x1", "copied_x2", "copied_y1", "copied_y2"])
df_bbox_copy.to_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy-3.pkl")

# %%
pd.set_option('display.max_colwidth', None)
dm = pd.read_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy-3.pkl")
print(len(dm))
dm.head()

#%%
# def draw_compare_image(df_copy):
#     root_images_PNG = "/home/single3/tintrung/brain-mri-tumor-images-PNG"
    

#     for index, row in df_copy.iterrows():
#         fig, axs = plt.subplots(1, 2, sharex=True, sharey= True)
#         fig.set_size_inches(15, 15)
#         copy_im = cv2.imread(os.path.join(root_images_PNG, row["imageUid"] + ".png"))
#         anot_im = cv2.imread(os.path.join(root_images_PNG, row["anot_image_uid"] + ".png"))
#         copy_im =  cv2.resize(copy_im, (anot_im.shape[0], anot_im.shape[1]) )
#         print(copy_im.shape)
#         print(anot_im.shape)
#         bbox_anot = convert_bbox(row["x1_anot"], row["x2_anot"], row["y1_anot"], row["y2_anot"])
#         rect_anot = patches.Rectangle((bbox_anot[0], bbox_anot[1]), bbox_anot[2], bbox_anot[3], linewidth=1, edgecolor='r', facecolor='none')


#         axs[0].imshow(copy_im)
#         axs[1].imshow(anot_im)
#         axs[1].add_patch(rect_anot)
#         # plt.savefig(os.path.join(test_src_folder, "test_image", str(index)))

#     # plt.subplots_adjust(wspace=0.025, hspace=0.025)

# draw_compare_image(dm)


# # %%

# # Split dcm file
# import shutil

# path_source = '/home/single3/tintrung/brain-mri-tumor-images-bboxes-copy_2'
# path_target_root = '/home/single3/tintrung/brain-mri-tumor-images-bboxes-copy_2_split'
# df_bboxes_copy = pd.read_pickle('/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl')

# for idx, row in tqdm(df_bboxes_copy.iterrows()):
#     path_src_file = os.path.join(path_source, row['imageUid']+'.png')
#     path_tar_studies = os.path.join(path_target_root, row['StudyInstanceUID'])
#     if not os.path.exists(path_tar_studies):
#         os.mkdir(path_tar_studies)
#     path_tar_series = os.path.join(path_tar_studies, row['SeriesInstanceUID'])
#     if not os.path.exists(path_tar_series):
#         os.mkdir(path_tar_series)
#     path_tar_file = os.path.join(path_tar_series, row['imageUid']+'.png')
#     shutil.copy(path_src_file, path_tar_file)



# #%%%%

# pd.set_option('display.max_colwidth', None)
# dm = pd.read_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy_4k_add.pkl")
# print(len(dm))
# dm.head()
