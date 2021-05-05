# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import numpy as np 
import pandas as pd
import pandas_read_xml as pdx

#%%

def process_xml(xml_path):
    """
    Process xml annotation file
    input: directory to xml file
    output: return required field in a list. The following fields in added in addition to field in process_label function: 
    date, patientid, patientPid, diagnosis

    NOTE: role of diagnosis column is unclear
    """
    
    # read xml file
    df = pdx.read_xml(xml_path, ["study_query"])
    df = df["study"][0]
    # extract date
    date = df["@date"]

    # extract patientid
    patientid = df["patient"]["@id"]
    patientPid = df["patient"]["@pid"]

    #extract diagnosis for each tag
    diagnosis = [ele["id"] for ele in df["diagnosis"]]  

    # extract every other info
    out = [process_label(label) for label in df["label"]]

    rows = []
    # append annotations
    for i in out:
        row = [date, patientid, patientPid, diagnosis] + i
        rows.append(row)

    return rows



#%%
path = "/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/labels.xml"
a = process_xml(path)
df_out = pd.DataFrame(a, columns = ["date", "patientid", "patientPid", "diagnosis","sessionid", "type_anot", "annotation", "scope", "unit", "timestamp", "imageuid", "seriesuid", "studyuid", "tag", "points"])
df_out.to_pickle("test_study.pkl")

#%%
def process_label(label):
    """
    Process each <label> tag inside each xml file
    input: <label> tag in form of orderdict
    output: extract the following fields: 
    sessionid, type_anot, annotation, scope, unit, timestamp, imageuid, seriesuid, studyuid, tag, points

    NOTE: each xml file has several <label> tag
    """

    sessionid = label["@sessionId"]
    type_anot = label["@type"]
    annotation = label["@annotation"]
    scope = label["@scope"]
    unit = label["@pointUnit"]
    timestamp = label["@createTimestamp"]
    imageuid = label["@imageUid"]
    seriesuid = label["@seriesUid"]
    studyuid = label["@studyUid"]

    #NOTE some file has noisy <tag>
    # only keep tumor relate to brain
    

    if len(label["tags"]["value"]) > 1:
        tag = []
        for sub_tag in label["tags"]["value"]:
            tag.append(sub_tag["@name"])
    else:
        tag = [label["tags"]["value"]["@name"]]
    
    # if type_anot == "global":
    #     print("Before", tag)
        # tag = list(set(tag) & check_label)
    #     print("After", tag)
    #     if (len(tag) > 1) & ("Other tumor" in tag) :
    #         print(tag, "Type 1")
    #     elif (len(tag) > 1) & ("Other tumor" not in tag):
    #         print(tag, "Type 2")

    #NOTE file 95, 111, 120 has error value in <point>
    if label["point"] != None:
        if len(label["point"]["value"]) == 12:
            points = [[float(point["@x"]), float(point["@y"]), float(point["@z"])] for point in label["point"]["value"]]
        else: 
            points = None
    else:
        points = None

    return [sessionid, type_anot, annotation, scope, unit, timestamp, imageuid, seriesuid, studyuid, tag, points]
























# %%
df = pd.read_pickle("/home/single3/Documents/tintrung/brainmri/trungdc/1.2.840.113619.6.408.217046082646534226129609417598798276891/test_dicom.pkl")
df.head()
# %%

