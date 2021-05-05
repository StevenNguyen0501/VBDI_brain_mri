#%%

# import library
import os
import pickle
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut

#%%
# set global variable
root = "//media/tungthanhlee/DATA/brain-mri-tumor-dicom-masked"
summary_dicom_path = "summary_dicom.pkl"
summary_anot_path = "summary_anot.pkl"



#%%
# helper function

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

def process_points(xy3d, xy_root, pixel_spacing):
    """
    Function convert a set of points in 3d plane into 2 plane
    Input:
    - xy3d: (x,y) with respect to a 3d plane
    - xyroot: (x,y) with respect to a 3d plane
    Output:
    NOT RETURN: xy2d: (x,y) with respect to xyroot in 2d plane
    - plot_point: bounding boxes (x,y,w,h) for convenience to plot with function Rectangle and to object detection model later
    """

    delta = np.sqrt(2*((pixel_spacing/2)**2))

    xy2d = []
    cur_im = []
    count1 = 1
    count2 = 0
    while count1 <  len(xy3d)+1:
        new_p = [(xy3d[count1-1][0] - xy_root[count2][0])/pixel_spacing,(xy3d[count1-1][1] - xy_root[count2][1])/pixel_spacing]
        # xy2d.append(new_p)
        cur_im.append(new_p)
        # print(new_p)
        if count1 % 4 == 0:
            xy2d.append(cur_im)
            cur_im = []
            count2 += 1
        count1 += 1

    plot_point = []
    for cur_im in xy2d:
        x = [ p[0] for p in cur_im]
        y = [ p[1] for p in cur_im ]
        bottomleftx = min(x)
        bottomlefty = min(y)
        height = max(y) - min(y)
        width = max(x) - min(x)
        out = [bottomleftx, bottomlefty, width, height]
        plot_point.append(out)
    return plot_point

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

def three_phrase_vis(series_uid, summary_dicom_path, summary_anot_path, plot = False, save_plot = False):
    """
    
    """
    # read in dataframe
    df1 = pd.read_pickle(summary_dicom_path)
    df2 = pd.read_pickle(summary_anot_path)

    # extract preferred information
    df1_extract = df1[df1.SeriesInstanceUID == chosen_series_uid].sort_values(by = ["SliceLocation"])

    # z_meta_slice = [float(value) for key,value in enumerate(df1_extract["SliceLocation"])]
    z_meta_imgpose = sorted([float(value[2])  for key,value in enumerate(df1_extract["ImagePositionPatient"])])

    points = np.array(list(df2[df2.seriesuid == chosen_series_uid]["points"])[0])
    

    # extract and check study uid
    study_uids = df1_extract["StudyInstanceUID"].drop_duplicates()
    
    if len(study_uids) > 1:
        print("ERROR: multiple studyuid")
        print(f"StudyUIDS: {study_uids}")
    else:
        print(f"StudyUID: {list(study_uids.values)[0]}")
    study_uid = study_uids.values[0]

    # extract slice index
    match_slice_index = [] # each element is [slice_index, error when matching]
    for z in np.unique(points[:, 2]):
        abs_store = []
        abs_store = [[index,np.abs(z - z_new)] for index,z_new in enumerate(z_meta_imgpose) if np.sign(z) == np.sign(z_new)]
        abs_store =  sorted(abs_store, key=lambda x:x[1])
        match_slice_index.append(abs_store[0])
        # print(f"Z in xml: {z} is matched with error {abs_store[0][1]} at slice index {abs_store[0][0]}")
    # print(f"Slice index and error:")
    # print(*match_slice_index, sep="\n")
    # print("\n")

    # extract further info for processing points
    study_folder = os.path.join(root, study_uid)
    image_paths = []
    imagepose = []
    for i in match_slice_index:
        image_paths.append( os.path.join(study_folder, df1_extract.iloc[i[0]]["NameFileXML"]))
        imagepose.append(df1_extract.iloc[i[0]]["ImagePositionPatient"])
    
    # print("Image path:")
    # print(*image_paths, sep="\n")
    # print("\n")

    pixel_spacing = float(df1_extract.iloc[0]["PixelSpacing"][0])
    bboxes = process_points([[point[0], point[1]] for point in points], [[float(p[0]) ,float(p[1])] for p in imagepose],pixel_spacing)
    
    # print("Point after process:")
    # print(*bboxes, sep="\n")
    # print("\n")

    if plot:
        imgs = []
        for i in image_paths:
            imgs.append(read_dicom(i))
        figure = plot_images(imgs, bboxes, np.unique(points[:, 2]), match_slice_index)
        figure.show()
    if save_plot:
        figure.savefig(f"test_im/{study_uid}-{series_uid}.jpg")
        
    return [series_uid, study_uid, image_paths, bboxes, match_slice_index]


#%%

df2 = pd.read_pickle(summary_anot_path)
path = []
for key,value in enumerate(df2["seriesuid"]):
    path.append(value)
path = pd.Series(path).drop_duplicates()
# print(list(path))


#%%

error_path = []
error_index = []
outs  = []
for index in range(len(path)):
    chosen_series_uid = list(path)[index]
    try:
        print("Series UID:",chosen_series_uid)
        out = three_phrase_vis(chosen_series_uid, summary_dicom_path, summary_anot_path, plot = False, save_plot = False)
        outs.append(out)
        print("___________SUCCESS!!!!!!!!!______________")
    except:
        print("Series UID:",chosen_series_uid)
        error_path.append(chosen_series_uid)
        error_index.append(index)
        print("___________ERROR!!!!!!!!!______________")
    print("#"*20)

#%%
print(len(error_index))

pd.set_option('display.max_colwidth', None)
df = pd.DataFrame(outs, columns = ["series_uid", "study_uid", "image_paths", "bboxes", "slice_index & error"])
df.head()





