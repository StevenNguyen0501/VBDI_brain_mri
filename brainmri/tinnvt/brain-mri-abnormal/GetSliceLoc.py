import os
import numpy as np 
import xml.etree.ElementTree as ET
import pydicom
import pandas as pd 
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

XML_DIR = "/media/datnt/data/medical-image-data/brain-mri-xml/1511-xml"
DCM_DIR = "/media/datnt/data/medical-image-data/brain-mri-dicom-filtered"

def get_loc(studyUid):
    # make label path (assuming we are working on 1511 filtered cases)
    label_path = os.path.join(XML_DIR, studyUid, "labels.xml")
    
    # load xml
    tree=ET.parse(label_path)
    root = tree.getroot()
    study = root.find('study')
    seriesUid = None
    
    # output list for this study
    positions = []    
    SOPInstanceUIDs = []
    
    # iterate through all labels in a xml file
    for label in study.iter('label'):
        assert studyUid == label.attrib['studyUid']
        if not seriesUid:
            seriesUid = label.attrib['seriesUid']
        else:
            assert seriesUid == label.attrib['seriesUid']

    for x in Path('{}/{}/DICOM'.format(DCM_DIR, studyUid)).iterdir():
        # assert DICOM file
        if not x.is_file():
            continue                    
        if '.dcm' not in x.name and '.dicom' not in x.name:
            continue
        
        dicom_path = '{}/{}/DICOM/{}'.format(DCM_DIR, studyUid, x.name)
        f = pydicom.dcmread(dicom_path)
        
        assert studyUid == f.StudyInstanceUID
        # only read instance from series in xml file
        if seriesUid != f.SeriesInstanceUID:
            continue
        # append to make list of instance UID and corresponding origin position
        positions.append(f.ImagePositionPatient[-1])
        SOPInstanceUIDs.append(f.SOPInstanceUID)
    
    SOPInstanceUIDs = np.array(SOPInstanceUIDs)
    positions = np.array(positions)
    idxs = np.argsort(positions)
    SOPInstanceUIDs = SOPInstanceUIDs[idxs]

    study_out = [[studyUid, seriesUid, s, p] for s,p in zip(SOPInstanceUIDs, range(len(positions)))]
    study_out = pd.DataFrame(data=study_out, columns=["studyUid", "seriesUid", "imageUid", "loc"])

    return study_out

if __name__ == "__main__":
    _1511_studies = os.listdir(XML_DIR)
    df = pd.DataFrame()
    for study_i in tqdm(_1511_studies):
        try:
            df_i = get_loc(study_i)
            df = pd.concat([df, df_i])
        except:
            print(study_i)
    df.to_csv("brain-mri-sops.csv", index=False)    