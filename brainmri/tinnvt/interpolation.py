# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import pickle
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut


# %%
lst_points = [[11, [[-33.32080078125, 13.382699966431, -20.75119972229],
                [-16.425800323486, 13.382699966431, -20.75119972229],
                [-16.425800323486, 33.547698974609, -20.75119972229],
                [-33.32080078125, 33.547698974609, -20.75119972229]]],
              [13, [[-33.32080078125, 13.382699966431, -6.3511900901794],
                [-16.425800323486, 13.382699966431, -6.3511900901794],
                [-16.425800323486, 33.547698974609, -6.3511900901794],
                [-33.32080078125, 33.547698974609, -6.3511900901794]]],
              [17, [[-40.405799865723, -32.397399902344, 15.24880027771],
                [56.059299468994, -32.397399902344, 15.24880027771],
                [56.059299468994, 65.15779876709, 15.24880027771],
                [-40.405799865723, 65.15779876709, 15.24880027771]]]]


# %%
pixel_spacing = 0.4492


# %%
listImagePositionPatient = [[-107.387, -106.245, -85.5513],
                            [-107.794, -106.074, -79.5675],
                            [-108.2, -105.904, -73.5837],
                            [-108.607, -105.733, -67.6],
                            [-109.014, -105.562, -61.6162],
                            [-109.421, -105.391, -55.6324],
                            [-109.828, -105.22, -49.6487],
                            [-110.234, -105.049, -43.6649],
                            [-110.641, -104.879, -37.6811],
                            [-111.048, -104.708, -31.6974],
                            [-111.455, -104.537, -25.7136],
                            [-111.862, -104.366, -19.7298],
                            [-112.268, -104.195, -13.7461],
                            [-112.675, -104.025, -7.76232],
                            [-113.082, -103.854, -1.77855],
                            [-113.489, -103.683, 4.20521],
                            [-113.896, -103.512, 10.189],
                            [-114.302, -103.341, 16.1727],
                            [-114.709, -103.17, 22.1565],
                            [-115.116, -103, 28.1403],
                            [-115.523, -102.829, 34.124],
                            [-115.93, -102.658, 40.1078],
                            [-116.336, -102.487, 46.0916],
                            [-116.743, -102.316, 52.0753],
                            [-117.15, -102.146, 58.0591]]


# %%
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
def find_missing_slices(listPoints3D_3Planes, listImagePositionPatient, fixelSpacing):
    """
    Find missing slices between 3 given slices
    Input: 
    - listPoints3D_3Planes: List of index slices and 3D-points in these slices [[index1, [list4Points1]],[index2, [list4Points2]], [index3, [list4Points3]]]
    - listImagePositionPatient:
    - fixelSpacing: 
    Output:
    - listSlices: List of list target index slices and 3D-points in these slices [[index1_, [list4Points1_]],[index2_, [list4Points2_]], [index3_, [list4Points3_]],...]
    """
    # Slices Upper, Middle, Lower
    list_index_slices_up_mid_low = [item[0] for item in listPoints3D_3Planes]
    listRootPointsUpMidLow = [listImagePositionPatient[idx] for idx in list_index_slices_up_mid_low]
    points3D = [item[1] for item in listPoints3D_3Planes]
    upper4Points = points3D[0]
    middle4Points = points3D[1]
    lower4Points = points3D[2]

    slicesUpMidLow = [[list_index_slices_up_mid_low[i], 
                    covert_coordinate(points3D[i], listRootPointsUpMidLow[i], fixelSpacing)] for i in range(len(list_index_slices_up_mid_low))]

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


# %%
# Test functions
target_plot_planes = find_missing_slices(lst_points, listImagePositionPatient, pixel_spacing)
target_plot_planes


# %%



# %%
list_index_slices_up_mid_low = [item[0] for item in lst_points]

listRootPoints = [listImagePositionPatient[idx] for idx in list_index_slices_up_mid_low]
RootPointsUpper = listRootPoints[0]
RootPointsMiddle = listRootPoints[1]
RootPointsLower = listRootPoints[2]

points = [item[1] for item in lst_points]
upper4Points = points[0]
middle4Points = points[1]
lower4Points = points[2]

slicesUpMidLow = [[list_index_slices_up_mid_low[i], 
                    covert_coordinate(points[i], listRootPoints[i], pixel_spacing)] for i in range(len(lst_points))]
slicesUpMidLow


# %%
index_upper_slice = list_index_slices_up_mid_low[0]
index_middle_slice = list_index_slices_up_mid_low[1]
index_lower_slice = list_index_slices_up_mid_low[2]

# Loop each plane
list_missing_slices = [] 
for index1 in range(index_upper_slice+1, index_middle_slice, 1):
    # print(index1)
    missing_slice_z_axis = listImagePositionPatient[index1][2]
    # print(missing_slice_z_axis)
    # Loop each point in set 4-points
    points_real_world_coordinates_in_slices = [find_point_in_slice(upper4Points[i], middle4Points[i], missing_slice_z_axis) for i in range(len(middle4Points))]
    points_in_slices = covert_coordinate(
                    listPoints3D=points_real_world_coordinates_in_slices,
                    pointRoot3D=listImagePositionPatient[index1],
                    pixelSpacing=pixel_spacing
                    )
    list_missing_slices.append([index1, points_in_slices])

print(list_missing_slices)


# %%
for index2 in range(index_middle_slice+1, index_lower_slice, 1):
    # print(index2)
    missing_slice_z_axis = listImagePositionPatient[index2][2]
    # print(missing_slice_z_axis)
    # Loop each point in set 4-points
    points_real_world_coordinates_in_slices = [find_point_in_slice(middle4Points[i], lower4Points[i], missing_slice_z_axis) for i in range(len(middle4Points))]
    points_in_slices = covert_coordinate(
                    listPoints3D=points_real_world_coordinates_in_slices,
                    pointRoot3D=listImagePositionPatient[index2],
                    pixelSpacing=pixel_spacing
                    )
    list_missing_slices.append([index2, points_in_slices])

print(list_missing_slices)


# %%
# Append 3 slices Upper, Middle, Lower 
list_missing_slices.extend(slicesUpMidLow)
list_missing_slices


# %%



# %%



# %%



# %%
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


# %%



