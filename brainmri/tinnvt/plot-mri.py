# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import time
import pickle5 as pickle
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut


# %%
df_dicom = pd.read_pickle('/home/single1/BACKUP/tintrung/brainmri/tinnvt/summary_dicom.pkl')
pd.set_option("display.max_colwidth", None)
df_dicom.head()


# %%
df_xml = pd.read_pickle('/home/single1/BACKUP/tintrung/brainmri/tinnvt/summary_xml.pkl')
pd.set_option("display.max_colwidth", None)
df_xml.head()


# %%
list_seriesUid_xml = []
for key,value in enumerate(df_xml['seriesUid']):
    list_seriesUid_xml.append(value)
set_seriesUid_xml = set(list_seriesUid_xml)
len(set_seriesUid_xml)


# %%
# Choice random index of list seriesUID xml, then map with dicom files have same seriesUID, sort values by "Slice Location"
# index = 113
# choisen_seriesUID = list(set_seriesUid_xml)[index]
choisen_seriesUID = '1.2.840.113619.2.388.57473.14165493.12404.1597274161.462'
df_one_seriesUID = df_dicom[df_dicom['SeriesInstanceUID']==choisen_seriesUID].sort_values(by='SliceLocation')
df_one_seriesUID = df_one_seriesUID.reset_index()
df_one_seriesUID


# %%
# Find dicom files have same seriesUID
df_get_point = df_xml[df_xml['seriesUid']==choisen_seriesUID]
df_get_point = df_get_point.reset_index()
df_get_point

# %% [markdown]
# Nhận xét: 
# 
# 1. Có những seriesUID có các bộ point 8 điểm (2 slices), thay vì 12 điểm (3 slices): 1.2.840.113619.2.410.15512023.5814788.18193.1579133498.587
# 
# 2. seriesUID có 1 bộ point 12 điểm
# 
# 3. seriesUID có 2 bộ point 12 điểm
# 
# 4. Có nhiều seriesUID, slice sắp xếp không đúng thứ tự
# 
# 
# 

# %%
def covert_coordinate(listPoints3D, pointRoot3D, pixelSpacing):
    """
    Function convert real coordiantes points 3D into point 2D
    Input:
    @param: listPoints3D: List of list 4 points real world coordinates [[x0,y0,z0],[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
    @param: pointRoot3D: point 3D ImagePositionPatient [x,y,z]
    @param: pixelSpacing
    Output:
    @param: plot_point: bounding boxes (x,y,w,h) for convenience to plot with function Rectangle and to object detection model later
    """
    # delta = np.sqrt(2*((pixelSpacing/2)**2))
    # pointRoot3D[0] = pointRoot3D[0] + delta
    # pointRoot3D[1] = pointRoot3D[1] + delta

    pt2D = []
    for pt in listPoints3D:
        new_pt = [(pt[0] - pointRoot3D[0])/pixelSpacing,
                  (pt[1] - pointRoot3D[1])/pixelSpacing]
        pt2D.append(new_pt)

    x = [p[0] for p in pt2D]
    y = [p[1] for p in pt2D]
    bottomleftx = min(x)
    bottomlefty = min(y)
    height = max(y) - min(y)
    width = max(x) - min(x)
    bbox = [bottomleftx, bottomlefty, width, height]
    return bbox



# %%
# Question: What another points which this points link to?  
points_one_seriesUID = df_get_point['point']
label_one_seriesUID = df_get_point[df_get_point['type']=='global']['tags'].values
# Maybe have more than one set of points, relative with more than one brain tumor 
# Get axis-z in each set points (xml files)
list_group_z_axis_sorted = []
list_point3d = []
dict_z_vs_point3d = {}

for set_points in points_one_seriesUID:
    if not set_points==['None']:
        sublist_z = []
        sublist_point = [set_points[n:n+4] for n in range(0,len(set_points),4)]
        list_point3d.append(sublist_point)

        for count, point in enumerate(set_points):
            z = point[2]
            sublist_z.append(z)
        sublist_z = list(set(sublist_z))
        sublist_z_sorted = sorted(sublist_z)
        list_group_z_axis_sorted.append(sublist_z_sorted)

print(list_group_z_axis_sorted)
print('*'*19)
print(len(list_point3d))

for i in range(len(list_point3d)):
    # TODO FIX BUG
    for j in range(len(list_point3d[0])):
        dict_z_vs_point3d[f'{list_point3d[i][j][0][2]}_{i}_{j}'] = list_point3d[i][j]

dict_z_vs_point3d


# %%



# %%



# %%
# # Get all coordinates of "Slice Location"
# z_meta_slice = [value for key, value in enumerate(df_one_seriesUID["SliceLocation"])]

# def min_distance(given_point: float, list_points: list):
#     """
#     Find a point of list point that has minimum distance with given point
#     """
#     list_distances = [np.abs(given_point - pt) for pt in list_points]
#     index_min = np.argmin(list_distances)
#     # print(list_distances)
#     target_point = float(list_points[index_min])
#     return [index_min, target_point]

# match_group_z = []
# for group_z_axis in list_group_z_axis: 
#     match_z = []
#     for each_z in group_z_axis:
#         new_item = min_distance(each_z, z_meta_slice)
#         # Truong hop 2 diem z deu nam gan 1 slice, quy uoc diem z thu hai se nam o slice tiep theo
#         if new_item in match_z:
#             new_item[0] = new_item[0] + 1
#             new_item[1] = float(z_meta_slice[new_item[0]])    
#         match_z.append(new_item)
#     match_group_z.append(match_z)
# print(match_group_z)


# %%
# Get coordinate-z of "Image Position Patient"
z_meta_imgpose = [float(value[2])  for key,value in enumerate(df_one_seriesUID["ImagePositionPatient"])]
pixelSpacing = float(df_one_seriesUID["PixelSpacing"][0][0])
# z_meta_imgpose = sorted(z_meta_imgpose)

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

# Get axis-z in each set points (xml files)
match_group_slices = []
for i, group_z_axis in enumerate(list_group_z_axis_sorted): 
    match_z = []
    for j, each_z in enumerate(group_z_axis):
        index_slice, z_coordinate = min_distance(each_z, z_meta_imgpose)
        # Truong hop 2 diem z deu nam gan 1 slice, quy uoc diem z thu hai se nam o slice tiep theo
        if (index_slice, z_coordinate) in match_z:
            seriesUID = df_one_seriesUID["SeriesInstanceUID"][index_slice+1]
            studiesUID = df_one_seriesUID["StudyInstanceUID"][index_slice+1]
            dcm_file = df_one_seriesUID["NameFileDCM"][index_slice+1]
            linked_pt = dict_z_vs_point3d[f'{each_z}_{i}_{j}']
            
            bbox = covert_coordinate(linked_pt, df_one_seriesUID["ImagePositionPatient"][index_slice+1], pixelSpacing)
            match_z.append([seriesUID, studiesUID, dcm_file, bbox, index_slice+1])
        else:
            seriesUID = df_one_seriesUID["SeriesInstanceUID"][index_slice]
            studiesUID = df_one_seriesUID["StudyInstanceUID"][index_slice]
            dcm_file = df_one_seriesUID["NameFileDCM"][index_slice]
            linked_pt = dict_z_vs_point3d[f'{each_z}_{i}_{j}']
            bbox = covert_coordinate(linked_pt, df_one_seriesUID["ImagePositionPatient"][index_slice], pixelSpacing)
            match_z.append([seriesUID, studiesUID, dcm_file, bbox, index_slice])
        # match_z: [index of slice in seriesUID dicom images, name of file dicom]
    match_group_slices.append(match_z)

print(len(match_group_slices))


# %%



# %%
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


# %%
def plot_images(imgs, annotations, slice_index):
    """
    Plot annotation into a series of images
    Input:
    - imgs: array-like pixel of image
    - annotation: 2D array of bounding boxes for corresponding images where each element is (x,y,w,h) - (bottomleftx, bottemlefty, width, height)
    Output:
    - Matplotlib figure of brain mri with bbounding boxes
    """
    assert(len(imgs) == len(annotations))

    fig, axes = plt.subplots(1,3)

    for i in range(3):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]), annotations[i][2], annotations[i][3], linewidth=1, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].imshow(imgs[i])
        axes[i].title.set_text(f'Slice {slice_index[i]}')
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        axes[i].figure.set_size_inches(15, 15)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.show()

    return fig

# %% [markdown]
# def process_one_seriesUID(df_dcm, df_anot, seriesUID):
#     
# 

# %%
# match_group_slices

ROOT_PATH = '/home/single1/BACKUP/tintrung/brain-mri-tumor-dicom-masked'

for i in range(len(match_group_slices)):
    dcm_paths = [os.path.join(ROOT_PATH, match_group_slices[i][j][1], match_group_slices[i][j][2]) for j in range(len(match_group_slices[i]))]
    # print(dcm_paths)
    imgs = [read_dicom(dcm_path) for dcm_path in dcm_paths]
    anots = [match_group_slices[i][j][3] for j in range(len(match_group_slices[i]))]
    # print(anots)
    # print(len(anots))
    # print(len(imgs))
    index_slice = [match_group_slices[i][j][4] for j in range(len(match_group_slices[i]))]
    plot_images(imgs, anots, index_slice)


# %%



# %%
# infile = open('/home/single1/BACKUP/tintrung/brainmri/tinnvt/fail_seriesUID.txt', 'r')
# fail_seriesUID = []
# for line in infile:
#     line = line.strip()
#     fail_seriesUID.append(line)
# fail_seriesUID = set(fail_seriesUID)

# success_seriesUID = set_seriesUid_xml - fail_seriesUID
# print(len(set_seriesUid_xml))
# print(len(fail_seriesUID))
# print(len(success_seriesUID))
# success_seriesUID
# success_seriesUID = list(success_seriesUID)
# succ_file = open('/home/single1/BACKUP/tintrung/brainmri/tinnvt/successful_seriesUID.txt', 'w')
# for i in range(len(success_seriesUID)):
#     succ_file.write(f'{success_seriesUID[i]}\n')

infile = open('/home/single1/BACKUP/tintrung/brainmri/tinnvt/fail_seriesUID.txt', 'r')
success_seriesUID = []
for line in infile:
    line = line.strip()
    success_seriesUID.append(line)
success_seriesUID = set(success_seriesUID)

# %% [markdown]
# {'1.2.840.113619.2.388.57473.14165493.12404.1597274161.342',
#  '1.2.840.113619.2.388.57473.14165493.12404.1597274161.462',
#  '1.2.840.113619.2.388.57473.14165493.12431.1595632873.131',
#  '1.2.840.113619.2.388.57473.14165493.12431.1595632874.21',
#  '1.2.840.113619.2.388.57473.14165493.12439.1601598609.243',
#  '1.2.840.113619.2.388.57473.14165493.12439.1601598609.7',
#  '1.2.840.113619.2.388.57473.14165493.12460.1599093184.246',
#  '1.2.840.113619.2.388.57473.14165493.12522.1602808835.537',
#  '1.2.840.113619.2.388.57473.14165493.12522.1602808835.643',
#  '1.2.840.113619.2.388.57473.14165493.12527.1602721925.407',
#  '1.2.840.113619.2.388.57473.14165493.12527.1602721925.481',
#  '1.2.840.113619.2.388.57473.14165493.12527.1602721925.498',
#  '1.2.840.113619.2.388.57473.14165493.12574.1596755745.171',
#  '1.2.840.113619.2.388.57473.14165493.12574.1596755745.196',
#  '1.2.840.113619.2.388.57473.14165493.12574.1596755745.263',
#  '1.2.840.113619.2.388.57473.14165493.12597.1594941312.645',
#  '1.2.840.113619.2.388.57473.14165493.12597.1594941312.856',
#  '1.2.840.113619.2.388.57473.14165493.12600.1595978241.202',
#  '1.2.840.113619.2.388.57473.14165493.12600.1595978241.85',
#  '1.2.840.113619.2.388.57473.14165493.12601.1600216232.253',
#  '1.2.840.113619.2.388.57473.14165493.12601.1600216232.636',
#  '1.2.840.113619.2.388.57473.14165493.12639.1596150914.295',
#  '1.2.840.113619.2.388.57473.14165493.12639.1596150914.570',
#  '1.2.840.113619.2.388.57473.14165493.12654.1599525247.480',
#  '1.2.840.113619.2.388.57473.14165493.12654.1599525247.526',

# %%
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

def bboxes_MRI_seriesUID(choisen_seriesUID, df_dicom, df_xml, ROOT_PATH_DCM_MASKED):
    """
    Return all bounding boxes and slices that have anotated
    Input:
    - choisen_seriesUID:
    - df_dicom:
    - df_xml:
    - ROOT_PATH_DCM_MASKED:
    Output:
    - List of list index slices and bounding boxes brain tumor in these slices
    """
    # Filter all xml files have same seriesUID, sort values by "Slice Location"
    df_one_seriesUID = df_dicom[df_dicom['SeriesInstanceUID']==choisen_seriesUID].sort_values(by='SliceLocation')
    df_one_seriesUID = df_one_seriesUID.reset_index()
    # Find dicom files have same seriesUID
    df_get_point = df_xml[df_xml['seriesUid']==choisen_seriesUID]
    df_get_point = df_get_point.reset_index()
    points_one_seriesUID = df_get_point['point']
    label_one_seriesUID = df_get_point[df_get_point['type']=='global']['tags'].values
    # Maybe have more than one set of points, relative with more than one brain tumor 
    # Get axis-z in each set points (xml files)
    list_group_z_axis_sorted = []
    list_point3d = []
    dict_z_vs_point3d = {}

    for set_points in points_one_seriesUID:
        if not set_points==['None']:
            sublist_z = []
            sublist_point = [set_points[n:n+4] for n in range(0,len(set_points),4)]
            list_point3d.append(sublist_point)

            for count, point in enumerate(set_points):
                z = point[2]
                sublist_z.append(z)
            sublist_z = list(set(sublist_z))
            sublist_z_sorted = sorted(sublist_z)
            list_group_z_axis_sorted.append(sublist_z_sorted)

    for i in range(len(list_point3d)):
        for j in range(len(list_point3d[0])):
            dict_z_vs_point3d[f'{list_point3d[i][j][0][2]}_{i}_{j}'] = list_point3d[i][j]

    # Get coordinate-z of "Image Position Patient"
    z_meta_imgpose = [float(value[2])  for key,value in enumerate(df_one_seriesUID["ImagePositionPatient"])]
    pixelSpacing = float(df_one_seriesUID["PixelSpacing"][0][0])
    # z_meta_imgpose = sorted(z_meta_imgpose)

    # Get axis-z in each set points (xml files)
    match_group_slices = []
    for i, group_z_axis in enumerate(list_group_z_axis_sorted): 
        match_z = []
        for j, each_z in enumerate(group_z_axis):
            index_slice, z_coordinate = min_distance(each_z, z_meta_imgpose)
            # Truong hop 2 diem z deu nam gan 1 slice, quy uoc diem z thu hai se nam o slice tiep theo
            if (index_slice, z_coordinate) in match_z:
                seriesUID = df_one_seriesUID["SeriesInstanceUID"][index_slice+1]
                studiesUID = df_one_seriesUID["StudyInstanceUID"][index_slice+1]
                dcm_file = df_one_seriesUID["NameFileDCM"][index_slice+1]
                linked_pt = dict_z_vs_point3d[f'{each_z}_{i}_{j}']
                
                bbox = covert_coordinate(linked_pt, df_one_seriesUID["ImagePositionPatient"][index_slice+1], pixelSpacing)
                match_z.append([seriesUID, studiesUID, dcm_file, bbox, index_slice+1])
            else:
                seriesUID = df_one_seriesUID["SeriesInstanceUID"][index_slice]
                studiesUID = df_one_seriesUID["StudyInstanceUID"][index_slice]
                dcm_file = df_one_seriesUID["NameFileDCM"][index_slice]
                linked_pt = dict_z_vs_point3d[f'{each_z}_{i}_{j}']
                bbox = covert_coordinate(linked_pt, df_one_seriesUID["ImagePositionPatient"][index_slice], pixelSpacing)
                match_z.append([seriesUID, studiesUID, dcm_file, bbox, index_slice])
                # match_z: [index of slice in seriesUID dicom images, name of file dicom]
        match_group_slices.append(match_z)

    # Plot all slices in this seriesUID
    for i in range(len(match_group_slices)):
        dcm_paths = [os.path.join(ROOT_PATH_DCM_MASKED, match_group_slices[i][j][1], match_group_slices[i][j][2]) for j in range(len(match_group_slices[i]))]
        imgs = [read_dicom(dcm_path) for dcm_path in dcm_paths]
        anots = [match_group_slices[i][j][3] for j in range(len(match_group_slices[i]))]
        index_slice = [match_group_slices[i][j][4] for j in range(len(match_group_slices[i]))]
        plot_images(imgs, anots, index_slice)

    return match_group_slices


# %%
slices = bboxes_MRI_seriesUID(choisen_seriesUID='1.2.840.113619.2.388.57473.14165493.12597.1594941312.645', 
                                    df_dicom=df_dicom, 
                                    df_xml=df_xml, 
                                    ROOT_PATH_DCM_MASKED=ROOT_PATH)


# %%
ROOT_PATH = '/home/single1/BACKUP/tintrung/brain-mri-tumor-dicom-masked'

for name_series in success_seriesUID:   
    try: 
        slices = bboxes_MRI_seriesUID(choisen_seriesUID=name_series, 
                                    df_dicom=df_dicom, 
                                    df_xml=df_xml, 
                                    ROOT_PATH_DCM_MASKED=ROOT_PATH)
    except Exception as error:
        print(name_series)
        print(error)
        print('*'*199)


# %%



