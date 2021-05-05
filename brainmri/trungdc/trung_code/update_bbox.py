#%%

import pandas as pd 
import numpy as np
import pydicom
import os
pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_colwidth', None)





#%%

bbox = pd.read_pickle("bbox.pkl")
bbox.head()


#%%

anot = pd.read_pickle("summary_anot.pkl")
anot = anot[anot.type_anot == "local"]
anot.head()
# %%

validate_index = anot["patientid"].drop_duplicates()
len(validate_index)

#%%
validate_index = anot["seriesuid"].drop_duplicates()
len(validate_index)
#%%
validate_index = anot["studyuid"].drop_duplicates()
len(validate_index)

# %%
dicom = pd.read_pickle("summary_dicom.pkl")
dicom.head()

# %%
# print(len(dicom["StudyInstanceUID"].drop_duplicates()))

print(len(dicom["SeriesInstanceUID"].drop_duplicates()))
# %%
imp = []
iop = []
studyuid = []
check = 0
for index, row in bbox.iterrows():
    extract_df = dicom[dicom.NameFileXML == row["DicomFileName"]]
    imp.append(extract_df["ImagePositionPatient"].values[0])
    iop.append(extract_df["ImageOrientationPatient"].values[0])
    if len(row["StudyUID"]) >0:
        studyuid.append(row["StudyUID"][0])
    else:
        studyuid.append(None)

    # print(extract_df["ImagePositionPatient"])
bbox["ImagePositionPatient"] = imp
bbox["ImageOrientationPatient"] = iop
bbox["StudyUID"] = studyuid
# %%

bbox.head()


#%%
bbox.to_pickle("bbox_update.pkl")





# %%
