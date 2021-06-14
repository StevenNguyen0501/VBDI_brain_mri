#%%
import pandas as pd
# from pandas.io.pickle import read_pickle

#%%
df = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_4k_add.pkl")
df.columns=  ['StudyInstanceUID', 'SeriesInstanceUID', 'imageUid_copied', 'label', 'x1_copied',
       'x2_copied', 'y1_copied', 'y2_copied', 'z', 'imageUid_based',
       'x1_based', 'x2_based', 'y1_based', 'y2_based']

df.head()

# %%
sequence = pd.read_csv("/home/single2/tintrung/VBDI_brain_mri/yolov5/df_series_clear.csv")
df_dicom = pd.merge(df, sequence, on=["SeriesInstanceUID", "StudyInstanceUID"])
len(df_dicom)
#%%
df = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_4k_add_sequence.pkl")
extract = df[df.SeriesLabel == "FLAIR"]
len(extract)

#%%
extract = df[df.SeriesLabel == "T1C"]
len(extract)

#%%

extract = df[df.SeriesLabel == "T2"]
len(extract)








# %%
old_df = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-dataset_2classes.pkl")
old_df.head()
old_df.columns
len(old_df[old_df.SequenceType == "FLAIR"])
# %%
df = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_4k_add_sequence.pkl")
df.head()
df.columns
extract = df[['StudyInstanceUID', 'SeriesInstanceUID', 'imageUid_copied', 'label',
       'x1_copied', 'x2_copied', 'y1_copied', 'y2_copied', 'SeriesLabel']].copy()
extract.columns = ['studyUid', 'seriesUid', 'imageUid', 'label', 'x1', 'x2', 'y1', 'y2',
       'SequenceType']
extract.head()
#%%
combineoldnew = pd.concat([old_df, extract])
combineoldnew.head()
len(combineoldnew)
# %%
combineoldnew.to_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_5k_oldnew.pkl")
combineoldnew.head()
# %%
combineoldnew["SequenceType"].unique()

# %%
combineoldnew = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_5k_oldnew.pkl")
len(combineoldnew[combineoldnew.SequenceType == "FLAIR"])
# %%
new = {}
for value in combineoldnew["SequenceType"].values:
       print(type(value))
       # if type(value) == "str":
       #        if value not in new.values():
       #               new[value] = 0
       #        else:
       #               new[value] += 1
       # else:
       #        print(value)
type(new[0])








# %%
