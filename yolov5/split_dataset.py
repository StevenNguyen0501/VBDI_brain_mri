import os
import cv2
import pickle
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from itertools import combinations
  
path_dataframe = '/home/single3/Documents/tintrung/brainmri/tinnvt/final_dataset.pkl'
path_train_yolov5 = '/home/single3/Documents/tintrung/yolov5_2classes/yolov5/data/train.txt'
path_val_yolov5 = '/home/single3/Documents/tintrung/yolov5_2classes/yolov5/data/val.txt'
path_root_images = '/home/single3/Documents/tintrung/yolov5_2classes/images'

df_final = pd.read_pickle(path_dataframe)
studiesUID_T1C = list(df_final[df_final['SeriesLabel']=='T1C']['StudyInstanceUID'].drop_duplicates())
N = int(len(studiesUID_T1C)*0.8)
comb_t1c = combinations(studiesUID_T1C, N)

for idx, group_studies_train in enumerate(list(comb_t1c)):
    
    group_studies_train = list(group_studies_train) 
    group_studies_val = list(set(studiesUID_T1C)-set(group_studies_train))
    
    dcm_t1c_train = []
    dcm_t1c_val = []

    num_train_imgs, num_train_mass_imgs, num_train_edema_imgs, num_val_imgs, num_val_mass_imgs, num_val_edema_imgs = 0, 0, 0, 0, 0, 0
   
    for stu in group_studies_train:
        lst_dcm = list(df_final[df_final['StudyInstanceUID']==stu]['DicomFileName'])
        dcm_t1c_train.extend(lst_dcm)
        num_train_imgs += len(lst_dcm)
        num_train_mass_imgs += len(list(df_final[(df_final['StudyInstanceUID']==stu) & (df_final['Tag']=='Mass_Nodule')]['DicomFileName']))
        num_train_edema_imgs += len(list(df_final[(df_final['StudyInstanceUID']==stu) & (df_final['Tag']=='Cerebral edema')]['DicomFileName'])) 
    
    for stu in group_studies_val:
        lst_dcm = list(df_final[df_final['StudyInstanceUID']==stu]['DicomFileName'])
        dcm_t1c_val.extend(lst_dcm)
        num_val_imgs += len(lst_dcm)
        num_val_mass_imgs += len(list(df_final[(df_final['StudyInstanceUID']==stu) & (df_final['Tag']=='Mass_Nodule')]['DicomFileName']))
        num_val_edema_imgs += len(list(df_final[(df_final['StudyInstanceUID']==stu) & (df_final['Tag']=='Cerebral edema')]['DicomFileName']))
    
    if num_train_imgs >= 6*num_val_imgs:
        if num_train_mass_imgs >= 6*num_val_mass_imgs:
            if num_train_edema_imgs >= 6*num_val_edema_imgs: 
                # Create file data/train.txt
                imgs_t1c_train = [os.path.join(path_root_images, item.replace('.dcm', '.png')) for item in dcm_t1c_train]
                imgs_t1c_train = map(lambda x: x+'\n', imgs_t1c_train)
                t1c_train = open(path_train_yolov5, 'w')
                t1c_train.writelines(imgs_t1c_train)
                t1c_train.close()
                # Create file data/val.txt
                imgs_t1c_val = [os.path.join(path_root_images, item.replace('.dcm', '.png')) for item in dcm_t1c_val]
                imgs_t1c_val = map(lambda x: x+'\n', imgs_t1c_val)
                t1c_val = open(path_val_yolov5, 'w')
                t1c_val.writelines(imgs_t1c_val)
                t1c_val.close()
