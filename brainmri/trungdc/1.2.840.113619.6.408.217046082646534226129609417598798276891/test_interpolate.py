#%%

# import library
import os
import pickle 
# import pickle
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut
    
#%%
# set global variable
root = "/home/single1/BACKUP/tintrung/brain-mri-tumor-dicom-masked"
summary_dicom_path = "/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/test_dicom.pkl"
summary_anot_path = "/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/test_study.pkl"

#%%

df_anot = pd.read_pickle(summary_anot_path)
df_anot = df_anot[df_anot.type_anot == "local"]


seriesuids = []
for key,value in enumerate(df_anot["seriesuid"]):
    seriesuids.append(value)
seriesuids = pd.Series(seriesuids).drop_duplicates()
print(f"Number of series to be process: {len(seriesuids)}")
df_anot.head()

#%%
df_dicom = pd.read_pickle(summary_dicom_path)
df_dicom =  df_dicom[df_dicom.SeriesInstanceUID =="1.2.840.113619.2.408.14196467.2124406.15025.1595462784.587"]
df_dicom
# df_dicom = df_dicom[df_dicom.SeriesDescription == "Ax DWI b=1000"]
# df_dicom.head()

#%%
def find_point_in_slice(point3D1, point3D2, z_axis):
    """
    Intersection point between plane and line in 3D-dimensions
    Input: 
    Line in 3d-dimensions 
    - point3D1: [x1,y1,z1]
    - point3D2: [x2,y2,z2]
    Plane in 3D-dimensions 
    - z_axis: z = z0
    Output:
    Intersection points
    """
    inter_x = (z_axis-point3D1[2])/(point3D2[2]-point3D1[2])*(point3D2[0]-point3D1[0]) + point3D1[0]
    inter_y = (z_axis-point3D1[2])/(point3D2[2]-point3D1[2])*(point3D2[1]-point3D1[1]) + point3D1[1]
    inter_point = [inter_x, inter_y, z_axis]
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

def find_missing_slices(listPoints3D, listImagePositionPatient, pixel_spacing):
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

#%%

def series_vis(series_uid, df_anot, df_dicom):

    print("#" *10)
    # each element has two part: points and labels
    points_labels = []
    for index, row in df_anot.iterrows():
        points_labels.append([row["points"], row["tag"]])
    print(f"Numer of points:  {len(points_labels)}")

    df_dicom = df_dicom.sort_values(by = ["SliceLocation"]).reset_index(drop = True)
    imp = df_dicom["ImagePositionPatient"]
    z_meta_imgpose = [value[2] for value in imp]
    filename = df_dicom["NameFileDCM"]

    for points in points_labels:
        points[0] = sorted(points[0], key=lambda x:x[2])

    
    study_uids = list(df_dicom["StudyInstanceUID"].drop_duplicates())


    match_slice_index = []
    for point in points_labels:
        cur_point_slide = []
        for z in np.unique([i[2] for i in point[0]]):
            abs_store = []
            abs_store = [[index,np.abs(z - z_new), z] for index,z_new in enumerate(z_meta_imgpose) if np.sign(z) == np.sign(z_new)]
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
            input1s.append([index,temp, tag, i ])
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

# %%

seriesuid = "1.2.840.113619.2.408.14196467.2124406.15025.1595462784.587"
out = series_vis(seriesuid, df_anot, df_dicom)
out = pd.DataFrame(out, columns = ["SliceIndex", "Bbox", "Tag","LesionIndex","DicomFileName", "SeriesUID", "StudyUID"])
out.head()
out.to_pickle("test_bbox.pkl")
# %%


]
# %%

df = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/test_bbox.pkl")
# %%
df
# %%
