#%%
import pandas as pd 
import tqdm
from tqdm import tqdm
from math import acos
from numpy import degrees
import numpy as np

import os
import cv2
import matplotlib.pyplot as plt
#%%
root_images_PNG = "/home/single3/tintrung/brain-mri-tumor-images-PNG"
test_src_folder = "/home/single3/tintrung/VBDI_brain_mri/brainmri/trungdc/copy_box_test"
pd.set_option('display.max_colwidth', None)
#%%
bbox = pd.read_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/trungdc/pkl_ver1/bbox_dat_extend.pkl")
print(bbox.columns)

#%%

dicom = pd.read_pickle("/home/single3/tintrung/VBDI_brain_mri/brainmri/trungdc/pkl_ver1/summary_dicom_extend.pkl")
dicom.head()
#%%
sequence = pd.read_csv("/home/single3/tintrung/VBDI_brain_mri/brainmri/trungdc/pkl_ver1/data_series_axial.csv")
sequence.head()


#%%

class Error(Exception):
    pass
class MultipleAnnotatedSeries(Error):
    def __init__(self, message):
        self.message = message

class MissingSequenceLabel(Error):
    def __init__(self, message):
        self.message = message


#%%
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
    if (given_point >= list_points[0]) & (given_point <= list_points[-1]):
        list_distances = [np.abs(given_point - pt) for pt in list_points]
        index = np.argmin(list_distances)
        target_point = float(list_points[index])
        error = abs(target_point - given_point)
        if error <= threshold:
            return [index, target_point, error]
        else:
            return None
    elif abs(given_point - list_points[0]) <= threshold:
        index = 0
        target_point = float(list_points[index])
        error = abs(target_point - given_point)
        return [index, target_point, error]
    elif abs(given_point - list_points[-1]) <= threshold:
        index = len(list_points)-1
        target_point = float(list_points[-1])
        error = abs(target_point - given_point)
        return [index, target_point, error]
    else:
        return None


#%%
studyuids = dicom["StudyInstanceUID"].drop_duplicates().values
chosen_study_uid = studyuids[20]
print(f"Chosen study uid: {chosen_study_uid}")

threshold = 1

# extract annotated seriesuid from given studyuid
annot_series_uid = bbox[bbox.studyUid == chosen_study_uid]["seriesUid"].drop_duplicates().values

# # check if there are multiple annotated series in a single study
if len(annot_series_uid) > 2:
    raise MultipleAnnotatedSeries("STUDY HAS MORE THAN ONE SERIES GOT ANNOTATED")
annot_series_uid = annot_series_uid[0]
print(f"Annotated series: {annot_series_uid}")

annot_series_sequence = sequence[sequence.SeriesInstanceUID == annot_series_uid]["SeriesLabelMerge"].values[0]
print(f"Sequence type of ANNOTATED series: {annot_series_sequence}")

# extract list of seriesuids to be copied from given studyuid
unprocessed_series = list(dicom[dicom.StudyInstanceUID == chosen_study_uid]["SeriesInstanceUID"].drop_duplicates().values)
unprocessed_series.remove(annot_series_uid)   # remove annotated series



# extract df of annotated series
anot_extract_df = bbox[bbox.seriesUid == annot_series_uid].copy()
anot_extract_df.sort_values(by = "z", inplace = True)
anot_extract_df.reset_index(inplace = True, drop = True)
print(f"Number of slices in annotated series: {len(anot_extract_df.index)}")


# checking iop of slices within annotated series
print("Checking iop between slices within annotated series")
max_diff = 0
anot_series_iop = anot_extract_df["ImageOrientationPatient"].drop_duplicates().values
# print(anot_series_iop)
if len(anot_series_iop) == 1:
    print("DONE")
else:
    flag = 0
    for column_index in range(len(anot_series_iop[0])):
        column = [float(i[column_index]) for i in anot_series_iop]        
        # print(column)
        diff = abs(degrees(max(column)) - degrees(min(column))) 
        # print(diff)
        if diff > max_diff:
            max_diff = diff
    print(f"Maximum difference between two slices: {max_diff}")

no_missing_label_series = 0
for series_uid in unprocessed_series:
    sequence_type = list(sequence[sequence.SeriesInstanceUID == series_uid]["SeriesLabelMerge"].values)

    # skip series that cannot classify sequence type
    if len(sequence_type) == 0:
        no_missing_label_series += 1
        # raise MissingSequenceLabel("Curent processing series has no sequence label")
        continue
    elif (sequence_type[0] != "t1") & (sequence_type[0] != "t2") & (sequence_type[0] != "flair"):
        continue


    print("#" * 10)
    sequence_type = sequence_type[0]
    print(f"Current processing series: {series_uid}")
    print(f"Sequence type: {sequence_type}")

    # extract z_value of processing series
    process_extract_df =  dicom[dicom.SeriesInstanceUID == series_uid].copy()
    process_extract_df.sort_values(by = ["z-axis-IMP"], inplace = True)
    process_extract_df.reset_index(inplace = True, drop = True)
    print(f"Number of slices before processing: {len(process_extract_df)}")


    # for each match slice in processing series
    # check for iop of that image with closest 
    output_box_copy = []
    for index, row in process_extract_df.iterrows():
        print("Slice DONE")
        match_z = min_distance(float(row["z-axis-IMP"]), list(anot_extract_df["z"]))
        output_box_copy_row = []
        
        if match_z == None:
            continue

        # checking difference between current slice and match slice
        processing_slice_iop = row["ImageOrientationPatient"]
        processing_slice_iop = [ acos(item) for item in processing_slice_iop]
        anot_slice_iop = anot_extract_df.iloc[match_z[0], -1]
        anot_slice_iop =  [ acos(item) for item in anot_slice_iop]

        diff_slice_iop = [ round(abs(processing_slice_iop[index] - anot_slice_iop[index]),3) for index in range(len(processing_slice_iop))]
        if sum(diff_slice_iop) > 2:
            continue
        
        # copy box from annotated slice to processing slice
        img_df_dcm = cv2.imread(os.path.join(root_images_PNG, row['DicomFileName'].replace('.dcm', '.png')))
        w1, h1, _ = img_df_dcm.shape
        chosen_row_anot = anot_extract_df.iloc[match_z[0]]

        output_box_copy_row.append(chosen_study_uid)
        output_box_copy_row.append(row['SeriesInstanceUID'])
        output_box_copy_row.append(row['DicomFileName'].split('.dcm')[0])
        output_box_copy_row.append(chosen_row_anot['label'])

        img_df_anot = cv2.imread(os.path.join(root_images_PNG, chosen_row_anot['imageUid']+'.png'))
        w2, h2, _ = img_df_anot.shape
        ratio = w1 / w2
        output_box_copy_row.append(chosen_row_anot['x1']*ratio)
        output_box_copy_row.append(chosen_row_anot['x2']*ratio)
        output_box_copy_row.append(chosen_row_anot['y1']*ratio)
        output_box_copy_row.append(chosen_row_anot['y2']*ratio)
        output_box_copy_row.append(chosen_row_anot['z'])
        output_box_copy_row.append(chosen_row_anot['imageUid'])
        output_box_copy_row.append(chosen_row_anot['x1'])
        output_box_copy_row.append(chosen_row_anot['x2'])
        output_box_copy_row.append(chosen_row_anot['y1'])
        output_box_copy_row.append(chosen_row_anot['y2'])

        output_box_copy.append(output_box_copy_row)
    
    out = pd.DataFrame(output_box_copy ,columns= ['studyUid', 'seriesUid', 'imageUid', 'label', 'x1', 'x2', 'y1', 'y2',
       'z', "anot_image_uid", "x1_anot", "x2_anot", "y1_anot", "y1_anot" ])
    out.to_pickle(os.path.join(test_src_folder, chosen_study_uid + ".pkl"  ))
        



    break



# %%

# %%
def draw_compare_image(df_copy):
    root_images_PNG = "/home/single3/tintrung/brain-mri-tumor-images-PNG"
    test_src_folder = "/home/single3/tintrung/VBDI_brain_mri/brainmri/trungdc/copy_box_test"

    for index, row in df_copy.iterrows():
        fig, axs = plt.subplots(1, 2, sharex=True, sharey= True)
        copy_im = cv2.imread(os.path.join(root_images_PNG, row["imageUid"] + ".png"))
        anot_im = cv2.imread(os.path.join(root_images_PNG, row["anot_image_uid"] + ".png"))
        axs[0].imshow(copy_im)
        axs[1].imshow(anot_im)

    plt.subplots_adjust(wspace=0.025, hspace=0.025)

#%%