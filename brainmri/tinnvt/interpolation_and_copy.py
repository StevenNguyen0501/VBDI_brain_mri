#%%
# import library
import os
# import pickle5 as pickle
import pickle
import pydicom
import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut
    
#%%
# set global variable
root = "/home/single3/Documents/tintrung/brain-mri-tumor-dicom-masked"
summary_dicom_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/summary_dicom.pkl"
summary_anot_path = "/home/single3/Documents/tintrung/brainmri/tinnvt/summary_anot.pkl"

#%%
df_anot = pd.read_pickle(summary_anot_path)
df_anot = df_anot[df_anot.type_anot == "local"].reset_index()
# df_anot: Add column "points", notice that in df_anot['points'] column have [] values
z_points = []
for lst_pts in df_anot['points']:
    z_subpts = []
    for idx, pt in enumerate(lst_pts):
        if idx % 4 == 0:
            z_subpts.append(pt[-1])
    z_points.append(z_subpts)
df_anot['z-axis-points'] = z_points

# Statistics the number of series to be processed
seriesuids = []
for key,value in enumerate(df_anot["seriesuid"]):
    seriesuids.append(value)
seriesuids = pd.Series(seriesuids).drop_duplicates()
print(f"Number of series to be processed: {len(seriesuids)}")
df_anot.columns

#%%
df_dicom = pd.read_pickle(summary_dicom_path)
# df_dicom: Add column "z-axis-ImagePositionPatient", notice that in df_dicom['ImagePositionPatient'] column have None values
z_axis = []
for item in list(df_dicom['ImagePositionPatient']):
    if item != None:
        z_axis.append(item[-1])
    else:
        z_axis.append(None)
df_dicom['z-axis-ImagePositionPatient'] = z_axis
df_dicom.columns

#%%
def find_point_in_slice(point3D1, point3D2, z_axis):
    """
    Intersection point between plane and line in 3D-dimensions
    Input: 
    - Line in 3d-dimensions 
        + point3D1: [x1,y1,z1]
        + point3D2: [x2,y2,z2]
    - Plane in 3D-dimensions 
        + z_axis: z = z0
    Output:
    Intersection points
    """
    inter_x = (z_axis - point3D1[2]) / (point3D2[2] - point3D1[2]) * (point3D2[0] - point3D1[0]) + point3D1[0]
    inter_y = (z_axis - point3D1[2]) / (point3D2[2] - point3D1[2]) * (point3D2[1] - point3D1[1]) + point3D1[1]
    inter_point = [int(inter_x), int(inter_y), z_axis]
    return inter_point

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
        new_pt = [(pt[0] - pointRoot3D[0]) / pixelSpacing,
                  (pt[1] - pointRoot3D[1]) / pixelSpacing]
        pt2D.append(new_pt)

    x = [p[0] for p in pt2D]
    y = [p[1] for p in pt2D]
    bottomleftx = min(x)
    bottomlefty = min(y)
    height = max(y) - min(y)
    width = max(x) - min(x)
    bbox = [bottomleftx, bottomlefty, width, height]
    return bbox

def interpolate_slices(listPoints3D, listImagePositionPatient, pixel_spacing):
    """
    Find missing slices between 3 given slices
    Input: 
    - listPoints3D: List of index slices and 3D-points in these slices [[index1, [list4Points1]],[index2, [list4Points2]], [index3, [list4Points3]]]
    - listImagePositionPatient:
    - pixelSpacing: 
    Output:
    - listSlices: List of list target index slices and 3D-points in these slices [[index1_, [list4Points1_]],[index2_, [list4Points2_]], [index3_, [list4Points3_]],...]
    """
    # Slices Upper, Middle, Lower
    list_index_slices_up_mid_low = [item[0] for item in listPoints3D]
    listRootPointsUpMidLow = [listImagePositionPatient[idx] for idx in list_index_slices_up_mid_low]
    points3D = [item[1] for item in listPoints3D]
    upper4Points = points3D[0]
    middle4Points = points3D[1]
    lower4Points = points3D[2]
    slicesUpMidLow = [[list_index_slices_up_mid_low[i], 
                    covert_coordinate(points3D[i], listRootPointsUpMidLow[i], pixel_spacing)] for i in range(len(list_index_slices_up_mid_low))]
    index_upper_slice = list_index_slices_up_mid_low[0]
    index_middle_slice = list_index_slices_up_mid_low[1]
    index_lower_slice = list_index_slices_up_mid_low[2]
    # Loop each plane Upper -> Middle 
    list_missing_slices = [] 
    for index1 in range(index_upper_slice+1, index_middle_slice, 1):
        missing_slice_z_axis = listImagePositionPatient[index1][2]
        # Loop each point in set 4-points
        points_real_world_coordinates_in_slices = [find_point_in_slice(upper4Points[i], middle4Points[i], missing_slice_z_axis) for i in range(len(middle4Points))]
        points_in_slices = covert_coordinate(
                        listPoints3D=points_real_world_coordinates_in_slices,
                        pointRoot3D=listImagePositionPatient[index1],
                        pixelSpacing=pixel_spacing
                        )
        list_missing_slices.append([index1, points_in_slices])
    # Loop each plane Middle -> Lower 
    for index2 in range(index_middle_slice+1, index_lower_slice, 1):
        missing_slice_z_axis = listImagePositionPatient[index2][2]
        # Loop each point in set 4-points
        points_real_world_coordinates_in_slices = [find_point_in_slice(middle4Points[i], lower4Points[i], missing_slice_z_axis) for i in range(len(middle4Points))]
        points_in_slices = covert_coordinate(
                        listPoints3D=points_real_world_coordinates_in_slices,
                        pointRoot3D=listImagePositionPatient[index2],
                        pixelSpacing=pixel_spacing
                        )
        list_missing_slices.append([index2, points_in_slices])
    # Append Upper, Middle, Lower slices
    list_missing_slices.extend(slicesUpMidLow)
    return list_missing_slices

#%%
def display_images(imgs, bboxes, labelNames, slice_index):
    """
    Plot annotation into a series of images
    Input:
    - imgs: array-like pixel of image
    - bboxes: 2D array of bounding boxes for corresponding images where each element is (x,y,w,h) - (bottomleftx, bottemlefty, width, height)
    - labelNames: 
    Output:
    - Matplotlib figure of brain mri with bbounding boxes
    """
    assert(len(imgs) == len(bboxes))
    fig, axes = plt.subplots(1,len(imgs), figsize=(19, 19))
    for i in range(len(imgs)):
        rect = patches.Rectangle((bboxes[i][0], bboxes[i][1]), bboxes[i][2], bboxes[i][3], linewidth=1, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].text(bboxes[i][0], bboxes[i][1], labelNames[i], fontsize=12, color='white')
        axes[i].imshow(imgs[i])
        axes[i].title.set_text(f'Slice {slice_index[i]}')
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        # axes[i].figure.set_size_inches(19, 19)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    return fig

def series_vis(series_uid, df_anot, df_dicom):

    print("#" *10)
    # each element has two part: points and labels
    points_labels = []
    for index, row in df_anot.iterrows():
        points_labels.append([row["points"], row["tag"]])
    print(f"Number of points:  {len(points_labels)}")

    df_dicom = df_dicom[df_dicom['SeriesInstanceUID']==series_uid].sort_values(by = ["SliceLocation"]).reset_index(drop = True)
    imp = df_dicom["ImagePositionPatient"]
    z_meta_imgpose = [value[2] for value in imp]
    filename = df_dicom["DicomFileName"]

    for points in points_labels:
        points[0] = sorted(points[0], key=lambda x:x[2])

    
    study_uids = list(df_dicom["StudyInstanceUID"].drop_duplicates())


    match_slice_index = []
    for point in points_labels:
        cur_point_slide = []
        for z in np.unique([i[2] for i in point[0]]):
            abs_store = []
            abs_store = [[index, np.abs(z - z_new), z] for index, z_new in enumerate(z_meta_imgpose) if np.sign(z) == np.sign(z_new)]
            abs_store =  sorted(abs_store, key=lambda x:x[1])
            cur_point_slide.append(abs_store[0][0])
            # print(f"Z in xml: {abs_store[0][2]} is matched with error {abs_store[0][1]} at slice index {abs_store[0][0]}")
        match_slice_index.append(cur_point_slide)
    # print(f"Slice index and error:")
    # print(*match_slice_index, sep="\n")

    for i in range(len(points_labels)):
        points_labels[i].append(match_slice_index[i])
    # print(*points_labels, sep="\n")
    # print("\n")
    pd.set_option("display.max_colwidth", None)

    input1s = []
    for i in range(len(points_labels)):
        cur_point = points_labels[i][0]
        if len(cur_point) % 4 != 0:
            print("ERROR in points: Points does not divide by 4")
        count = 0
        j = 0
        while count < len(cur_point):
            temp = cur_point[count:count+4]
            index = points_labels[i][2][j]
            tag = points_labels[i][1]
            input1s.append([index, temp, tag, i])
            count += 4
            j += 1
    
    # print("ORIGINAL DFFFF")
    df = pd.DataFrame(input1s, columns = ["Index", "Points", "Tag", "Group"])
    # print(df)

    # df = df.sort_values(by = ["Index"], axis = 0).reset_index(drop=True)
    # print(df)
        
    pixelSpacing = list(df_dicom["PixelSpacing"])[0]
    if len(pixelSpacing) > 1:
        pixelSpacing = pixelSpacing[0]


    unique_val = np.unique(df["Group"].values)
    outs = []
    for i in unique_val:
        # print("START")
        # print("DF FOR CUR POINT")
        curdf = df[df.Group == i]
        add_info = [[row["Tag"][0], row["Group"]] for index, row in curdf.iterrows()][0]
        # print(add_info)
        # print("ADD_INFO")
        # add_info = list(add_info)[0]
        # print(add_info)

        # print("INPUT FOR FUNCITON")
        # print(*curpoint, sep="\n")

        # print("OUTPUT FOR CUR POINT")

        out = find_missing_slices([[row["Index"], row["Points"]] for index, row in curdf.iterrows()], imp, float(pixelSpacing))
        # print(*out, sep="\n")
        for j in out:
            j.extend(add_info)
            j.append(filename[j[0]])
            j.append(series_uid)
            j.append(study_uids)

        outs.extend(out)
    outs = sorted(outs, key=lambda x:x[0])
    # print(*outs, sep="\n")

    return outs

#%% 
def plot_images(imgs, annotations, z, slice_index):
    """
    Plot annotation into a series of images
    Input:
    - imgs: array-like pixel of image
    - annotation: 2D array of bounding boxes for corresponding images where each element is (x, y, w, h) - (bottomleftx, bottemlefty, width, height)
    Output:
    - Matplotlib figure of brain mri with bbounding boxes
    """
    assert(len(imgs) == len(annotations))

    fig, axes = plt.subplots(1,3)

    for i in range(3):
        rect = patches.Rectangle((annotations[i][0], annotations[i][1]), annotations[i][2], annotations[i][3], linewidth=1, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].imshow(imgs[i])
        axes[i].title.set_text(f'Slice {slice_index[i][0]} with z = {z[i]:.2f}')
        axes[i].xaxis.set_visible(False)
        axes[i].yaxis.set_visible(False)
        axes[i].figure.set_size_inches(15, 15)
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    return fig

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
    df_get_point = df_xml[df_xml['seriesUid']==choisen_seriesUID].reset_index()
    points_one_seriesUID = df_get_point['point']
    # Find dicom files have same seriesUID
    df_one_seriesUID = df_dicom[df_dicom['SeriesInstanceUID']==choisen_seriesUID].sort_values(by='SliceLocation').reset_index()
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

    # Plot all slices in this seriesUID
    for i in range(len(match_group_slices)):
        dcm_paths = [os.path.join(ROOT_PATH, match_group_slices[i][j][1], match_group_slices[i][j][2]) for j in range(len(match_group_slices[i]))]
        imgs = [read_dicom(dcm_path) for dcm_path in dcm_paths]
        anots = [match_group_slices[i][j][3] for j in range(len(match_group_slices[i]))]
        index_slice = [match_group_slices[i][j][4] for j in range(len(match_group_slices[i]))]
        plot_images(imgs, anots, index_slice)

    return match_group_slices


#%%%
# # Plot images with old dataframe dataset.pkl
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import cv2
# from tqdm import tqdm

# dataset = pd.read_pickle('/home/single3/Documents/tintrung/brainmri/tinnvt/dataset.pkl')
# pathSrcImages = '/home/single3/Documents/tintrung/yolov5_2classes/images'
# pathSavedImages = '/home/single3/Documents/tintrung/yolov5_2classes/tmp_plot_imgs'

# list_DCMs = list(dataset['DicomFileName'].drop_duplicates())

# for idx in tqdm(range(len(list_DCMs))):
    
#     try:
#         nameDCM = list_DCMs[idx]
#         df_dcm = dataset[dataset['DicomFileName']==nameDCM].reset_index()
#         namePNG = nameDCM.replace('.dcm', '.png')
#         pathPNG = os.path.join(pathSrcImages, namePNG)
#         img = cv2.imread(pathPNG)
#         for i, labelName in enumerate(list(df_dcm['Tag'])):
#             if labelName=='Mass_Nodule':
#                 img = cv2.rectangle(img, (int(df_dcm['Bbox'][i][0]), int(df_dcm['Bbox'][i][1])), (int(df_dcm['Bbox'][i][0]+df_dcm['Bbox'][i][2]), int(df_dcm['Bbox'][i][1]+df_dcm['Bbox'][i][3])), (255, 0, 0), thickness=3, lineType=cv2.LINE_AA)
#                 img = cv2.putText(img, labelName, (int(df_dcm['Bbox'][i][0]), int(df_dcm['Bbox'][i][1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 0, 0], thickness=2, lineType=cv2.LINE_AA)
#             elif labelName=='Cerebral edema':
#                 img = cv2.rectangle(img, (int(df_dcm['Bbox'][i][0]), int(df_dcm['Bbox'][i][1])), (int(df_dcm['Bbox'][i][0]+df_dcm['Bbox'][i][2]), int(df_dcm['Bbox'][i][1]+df_dcm['Bbox'][i][3])), (125, 0, 255), thickness=3, lineType=cv2.LINE_AA)
#                 img = cv2.putText(img, labelName, (int(df_dcm['Bbox'][i][0]), int(df_dcm['Bbox'][i][1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 1, [125, 0, 255], thickness=2, lineType=cv2.LINE_AA)
#         cv2.imwrite(os.path.join(pathSavedImages, namePNG), img)
#     except Exception as error:
#         print(f'File {nameDCM} has error {error}')

# %%

