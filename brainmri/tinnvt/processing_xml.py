import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pandas_read_xml as pdx
from pydicom.pixel_data_handlers.util import apply_voi_lut


folder_label = "/media/tungthanhlee/DATA/brain-mri-tumor-xml"
patient_folder = [os.path.join(folder_label, i) for i in os.listdir(folder_label)]
labels_path = []
for subdir in patient_folder:
    for label_name in os.listdir(subdir):
        label_path = os.path.join(subdir, label_name)
        labels_path.append(label_path)
print("The number of total labels:", len(labels_path))
print(labels_path[0])


df = pdx.read_xml(labels_path[0], ["study_query"])

patientPid_list = []
diagnosis_list = []
sessionId_list = []
type_list = []
annotation_list = []
scope_list = []
pointUnit_list = []
createTimestamp_list = []
imageUid_list = []
seriesUid_list = []
studyUid_list = []
tags_list = []
point_list = []

print(df["study"]["label"])
for idx1 in range(len(df["study"]["label"])):
    obj = df["study"]["label"][idx1]
    # patientPid_list
    patientPid_list.append(df["study"]["@patientPid"])
    # diagnosis_list
    diagnosis_sub = []
    for idx2 in range(len(df["study"]["diagnosis"])):
        diagnosis_sub.append(df["study"]["diagnosis"][idx2]["id"])
    diagnosis_list.append(diagnosis_sub)
    # sessionId_list
    sessionId_list.append(obj["@sessionId"])
    # type_list
    type_list.append(obj["@type"])
    # annotation_list
    annotation_list.append(obj["@annotation"])
    # scope_list
    scope_list.append(obj["@scope"])
    # pointUnit_list
    pointUnit_list.append(obj["@pointUnit"])
    # createTimestamp
    createTimestamp_list.append(obj["@createTimestamp"])
    # imageUid_list 
    imageUid_list.append(obj["@imageUid"])
    # seriesUid_list
    seriesUid_list.append(obj["@seriesUid"])
    # studyUid_list
    studyUid_list.append(obj["@studyUid"])
    # tags_list
    tags_list.append(obj["tags"]["value"]["@name"])
    # point_list
    # print('aaaaa', obj["point"]["value"])
    # point_list.append(obj["point"]["value"])


print(len(patientPid_list))
print(len(diagnosis_list))
print(len(sessionId_list))
print(len(type_list))
print(len(annotation_list))
print(len(scope_list))
print(len(pointUnit_list))
print(len(createTimestamp_list))
print(len(imageUid_list))
print(len(seriesUid_list))
print(len(studyUid_list))
print(len(tags_list))
print(len(point_list))