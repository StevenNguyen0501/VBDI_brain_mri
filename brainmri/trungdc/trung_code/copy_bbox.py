#%%
import pandas as pd 
import tqdm
from tqdm import tqdm


#%%
bbox = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/pkl_ver1/bbox_dat_extend.pkl")
bbox.head()
#%%

dicom = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/pkl_ver1/summary_dicom_extend.pkl")
dicom.head()
#%%
sequence = pd.read_csv("/home/single3/Documents/tintrung/brainmri/trungdc/pkl_ver1/data_series_axial.csv")
sequence.head()
#%%

class Error(Exception):
    pass
class MultipleAnnotatedSeries(Error):
    def __init__(self, message):
        self.message = message

class MissingSequenceLabel(Error):
    def __init__(self, message):
        self.message = message
#%%
studyuids = dicom["StudyInstanceUID"].drop_duplicates().values
chosen_study_uid = studyuids[20]
print(f"Chosen study uid: {chosen_study_uid}")


# %%

# extract annotated seriesuid from given studyuid
annot_series = bbox[bbox.studyUid == chosen_study_uid]["seriesUid"].drop_duplicates().values

# # check if there are multiple annotated series in a single study
if len(annot_series) > 2:
    raise MultipleAnnotatedSeries("STUDY HAS MORE THAN ONE SERIES GOT ANNOTATED")
annot_series = annot_series[0]
print(f"Annotation series: {annot_series}")

annot_series_sequence = sequence[sequence.SeriesInstanceUID == annot_series]["SeriesLabelMerge"].values[0]
print(f"Sequence type of ANNOTATED series: {annot_series_sequence}")

# extract list of seriesuids to be copied from given studyuid
unprocessed_series = list(dicom[dicom.StudyInstanceUID == chosen_study_uid]["SeriesInstanceUID"].drop_duplicates().values)
unprocessed_series.remove(annot_series)   # remove annotated series

# extract z from annotated series
Z_anot =   sorted(bbox[bbox.seriesUid == annot_series]["z"].values)
print(f"Z_anot: {Z_anot}")


no_missing_label_series = 0
for series_uid in unprocessed_series:
    sequence_type = sequence[sequence.SeriesInstanceUID == series_uid]["SeriesLabelMerge"].values

    # skip series that cannot classify sequence type
    if len(sequence_type) == 0:
        no_missing_label_series += 1
        # raise MissingSequenceLabel("Curent processing series has no sequence label")
        continue
    print("#" * 10)
    sequence_type = sequence_type[0]
    print(f"Current processing series: {series_uid}")
    print(f"Sequence type: {sequence_type}")

    # extract z_value of processing series
    Z_compare = sorted(dicom[dicom.SeriesInstanceUID == series_uid]["z-axis-IMP"].values)
    # print(f"Z_compare: {Z_compare}")
    print(f"Number of slices before extracting: {len(Z_compare)}")


    threshold = 0.2

    # checking iop of slices within processing series 
    cur_series_iop = dicom[dicom.SeriesInstanceUID == series_uid]["ImageOrientationPatient"].drop_duplicates().values
    if len(cur_series_iop) == 1:
        print("Checking iop between slices withing curent processing series DONE")
    else:
        flag = 0
        for column in range(len(cur_series_iop[0])):
            cur_column  = [row[column] for row in cur_series_iop]
            diff  = max(cur_column) - min(cur_column)
            if diff >= threshold:
                flag = 1
                break
        if flag == 1:
            print(f"IOP between slices within processing series VARIES more than {threshold}")
        else:
            print("Checking iop between slices within curent processing series DONE")


    
    # checking iop of slices between annotated series and copy series
    anot_series_iop = dicom[dicom.SeriesInstanceUID == annot_series]["ImageOrientationPatient"].drop_duplicates().values
    if len(anot_series_iop) == 1:
        print("Checking iop between slices withing annotated processing series DONE")
    else:
        flag = 0
        for column in range(len(anot_series_iop[0])):
            diff  = max(cur_column) - min(cur_column)
            if diff >= threshold:
                flag = 1
                break
        if flag == 1:
            print(f"IOP between slices within annotated series VARIES more than {threshold}")
        else:
            print("Checking iop between slices within curent annotated series DONE")

    # checking iop between annotated series and processing series
    sample_procesing_iop = cur_series_iop[0]
    sample_annot_iop = anot_series_iop[0]
    for column in range(6):
        diff  = abs(sample_annot_iop[column] - sample_procesing_iop[column])
        print(diff)
        if diff >= threshold:
            flag = 1
            break
    if flag == 1:
        print(f"IOP between slices within annotated series VARIES more than {threshold}")
    else:
        print("Checking iop between slices within curent annotated series DONE")







# %%
