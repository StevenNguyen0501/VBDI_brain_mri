# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
import pickle
import pydicom
import pandas as pd
root_folder_path  = "/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/DICOM"


# %%
file_paths = []
folder_path = root_folder_path
for file_name in os.listdir(folder_path):
    file_paths.append(os.path.join(folder_path, file_name))
print(f"Total images: {len(file_paths)}")


# %%
# df = pd.read_pickle("summary_dicom.pkl")
# pd.set_option("display.max_colwidth", None)
# df.head()


# %%
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


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




# %%
height = []
width = []
for index, row in df.iterrows():
    cur = 0
    try:
        path = os.path.join(root_folder_path, row["StudyInstanceUID"])
        path = os.path.join(path, row["NameFileXML"])
        x = np.shape(read_xray(path))
        height.append(x[0])
        width.append(x[1])
    except:
        cur += 1


# %%
print(cur)


# %%
print(len(df))
print(len(height))


# %%



