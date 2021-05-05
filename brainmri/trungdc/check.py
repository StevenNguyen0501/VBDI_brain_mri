#%%
import pandas as pd
from tqdm import tqdm

#%%

df = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/brain-mri-xml-dataset.pkl")

df = df[(df.label == "Mass/Nodule") | (df.label == 'Cerebral edema')]
dicom_path = "/home/single3/Documents/tintrung/brainmri/trungdc/pkl_ver1/summary_dicom.pkl"
df_dicom = pd.read_pickle(dicom_path)

column_z = []
column_iop = []
for index, row in tqdm(df.iterrows()):
    origin_filename = row["imageUid"] + ".dcm"
    assert origin_filename in list(df_dicom["NameFileXML"].drop_duplicates())
    
    match_df = df_dicom[df_dicom.NameFileXML == origin_filename]
    assert len(match_df) == 1

    match_z = match_df["ImagePositionPatient"].values[0][-1]
    column_z.append(match_z)

    match_iop = match_df["ImageOrientationPatient"].values[0]
    column_iop.append(match_iop)

df["z"] = column_z
df["ImagePositionPatient"] = column_iop

# %%
df.to_pickle("")