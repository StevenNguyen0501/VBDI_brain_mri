#%%

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import pydicom
import os
from pydicom.pixel_data_handlers.util import apply_voi_lut
import matplotlib.patches as patches


#%%
pd.set_option("display.max_colwidth", None)
df = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/test_bbox.pkl")
df.head()


# %%
test_seriesuid = list(df["SeriesUID"].drop_duplicates())
cur_test = test_seriesuid[0]
print(f"Test seriesuid: {cur_test}")

#%%

def read_xray(path, voi_lut = True, fix_monochrome = True):
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
# df[df.SeriesUID == cur_test]

#%%


def plot_series_with_bbox(df):
    root = "/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/DICOM/"
    sliceIndex = list(df["SliceIndex"].drop_duplicates())
    print(sliceIndex)

    imgsname = list(df["DicomFileName"].drop_duplicates())
    print("Check lem", len(imgsname))

    numImgs=  len(sliceIndex)
    print(f"Num Images: {numImgs}")


    fig, axs = plt.subplots(numImgs, 3,  figsize=(19, 5*numImgs//3+1))
    plt.subplots_adjust(wspace=0, hspace=0)

    cur = 0
    for i in range(numImgs//3+1):
        for j in range(3):
            print(cur)
            if cur == numImgs:
                fig.savefig("test_im/sample.jpg")

            curdf = df[df.SliceIndex == sliceIndex[cur]]

            # axs[i,j].set_title(f"Slice {sliceIndex[cur]}", color = "white", fontweight='bold')
            print(sliceIndex[cur])

            bboxes = list(curdf["Bbox"])
            labels = list(curdf["Tag"])
            # print(bbox)

            for bbox,label in zip(bboxes, labels):
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
                axs[i,j].add_patch(rect)
                axs[i,j].text(bbox[0], bbox[1], label, fontsize=9, color='white')

            # study_name = list(curdf["StudyUID"])[0][0]
            path = os.listdir(root)[i]
            image_name = os.path.join(root, path)
            img = read_xray(image_name)
            axs[i,j].imshow(img)

            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)



            
            cur += 1


plot_series_with_bbox(df)




plot_series_with_bbox(df[df.SeriesUID == cur_test])
# %%
# x = df["DicomFileName"].value_counts()
# count = {}
# for i in x:
#     if i not in count:
#         count[i] = 1
#     else:
#         count[i] += 1
# a = pd.DataFrame(list(count.items()), columns=["Số box/ảnh", "Số ảnh"] )
# # a.set_index("Số box/ảnh", inplace=True)
# a.sort_values(by = "Số box/ảnh")


# %%
