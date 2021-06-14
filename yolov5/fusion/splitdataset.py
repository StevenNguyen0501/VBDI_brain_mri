#%%
import pandas as pd
from random import shuffle
import os


#%%
df_new = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_5k_oldnew.pkl")
studyuid = []
for sequence in ["FLAIR", "T1C", "T2"]:
    extract = list(df_new[df_new.SequenceType == sequence]["studyUid"].drop_duplicates())
    studyuid.extend(extract)
studyuid = list(set(studyuid))
print(len(set(studyuid)))
num_test_studyuid = 30
shuffle(studyuid)
test_study = studyuid[0:num_test_studyuid]
train_study = studyuid[num_test_studyuid:]
print(len(test_study))
print(len(train_study))

#increment file name every time shuffle
saved_split_dir = "/home/single2/tintrung/VBDI_brain_mri/yolov5/fusion/data_split"
pd.Series(train_study).to_pickle(os.path.join(saved_split_dir, "train_studyuid.pkl"))
pd.Series(test_study).to_pickle(os.path.join(saved_split_dir, "test_studyuid.pkl"))

# %%
df_final = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/brain-mri-xml-bboxes-copy_5k_oldnew.pkl")
train_study = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/fusion/data_split/train_studyuid.pkl")
test_study = pd.read_pickle("/home/single2/tintrung/VBDI_brain_mri/yolov5/fusion/data_split/test_studyuid.pkl")
# print(len(train_study))
# print(len(test_study))

# print(len(set(train_study)))
# print(len(set(test_study)))
# print(len(train_study.union((test_study))))
test_imageuid = []
for testuid in set(test_study):
    test_imageuid.extend(list(df_final[(df_final.studyUid == testuid) & ((df_final.SequenceType == "FLAIR"))  ]["imageUid"]))
test_imageuid = set(test_imageuid)

train_imageuid = []
for trainuid in set(train_study):
    train_imageuid.extend(list(df_final[(df_final.studyUid == trainuid) & ((df_final.SequenceType == "FLAIR") ) ]["imageUid"]))
train_imageuid = set(train_imageuid)
print(len(train_imageuid))
print(len(set(train_imageuid)))
print(len(test_imageuid))
print(len(set(test_imageuid)))
pd.Series(list(train_imageuid)).to_pickle(os.path.join(saved_split_dir, "train_imageuid.pkl"))
pd.Series(list(test_imageuid)).to_pickle(os.path.join(saved_split_dir, "test_imageuid.pkl"))


path_test_yolov5 = "/home/single2/tintrung/VBDI_brain_mri/yolov5/fusion/data_split/test.txt"
path_train_yolov5 = "/home/single2/tintrung/VBDI_brain_mri/yolov5/fusion/data_split/train.txt"

path_root_images = "/home/single2/tintrung/images"
testfile = open(path_test_yolov5, "w")
for test_name in test_imageuid:
    test_name += ".png"
    test_name += "\n"
    test_name = os.path.join(path_root_images, test_name)
    testfile.write(test_name)
testfile.close()

trainfile = open(path_train_yolov5, "w") 
for train_name in train_imageuid:
    train_name += ".png"
    train_name += "\n"
    train_name = os.path.join(path_root_images, train_name)
    trainfile.write(train_name)
trainfile.close()
# %%
