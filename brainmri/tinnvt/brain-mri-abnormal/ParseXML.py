import os
import numpy as np 
import xml.etree.ElementTree as ET
import pydicom
import pandas as pd 
from pathlib import Path
from tqdm import tqdm
import math

import warnings
warnings.filterwarnings("ignore")

XML_DIR = "/home/single3/Documents/tintrung/brain-mri-tumor-xml"
DCM_DIR = "/home/single3/Documents/tintrung/brain-mri-tumor-dicom-masked"

def prepare_image_process(studyUid):
    # make label path (assuming we are working on 1511 filtered cases)
    label_path = os.path.join(XML_DIR, studyUid, "labels.xml")  
    
    # load xml
    tree = ET.parse(label_path)
    root = tree.getroot()
    study = root.find('study')
    
    # output list for this study
    study_out = []
    
    # iterate through all labels in a xml file
    for label in study.iter('label'):
        # look at local labels only; all global diseases are treated as "abnormal"
        if label.attrib['type'] == 'local':
            # assert studyUid consistancy
            assert studyUid == label.attrib['studyUid']
            # read information from xml file
            seriesUid = label.attrib['seriesUid']
            tag = label.find('tags')
            point = label.find('point')
            # list of origin locations for instance in a series
            positions = []
            # list of instance UID in a series
            SOPInstanceUIDs = []
            # pixel spacing in instance
            pixelspacing = None
            # we assume that the DICOM dirs are located in "brain-mri-dicom-filtered" folder
            for x in Path('{}/{}'.format(DCM_DIR, studyUid)).iterdir():
                # assert DICOM file
                if not x.is_file():
                    continue                    
                if '.dcm' not in x.name and '.dicom' not in x.name:
                    continue
                
                dicom_path = '{}/{}/{}'.format(DCM_DIR, studyUid, x.name)
                f = pydicom.dcmread(dicom_path)
                
                assert studyUid == f.StudyInstanceUID
                # only read instance from series in xml file
                if seriesUid != f.SeriesInstanceUID:
                    continue

                if pixelspacing is None:
                    pixelspacing = f.PixelSpacing
                else:
                    # assert pixel spacing consistancy
                    assert pixelspacing == f.PixelSpacing
                # append to make list of instance UID and corresponding origin position
                positions.append(f.ImagePositionPatient)
                SOPInstanceUIDs.append(f.SOPInstanceUID)
            
            SOPInstanceUIDs = np.array(SOPInstanceUIDs)
            positions = np.array(positions)
            # get x, y origin for each slice
            comp = positions[np.argsort(positions[:,2])]            
            xmin, ymin = comp[0,:2]
            # now use position as list of z origin for each slice
            positions = positions[:,2]
            # sorting instance by z-axis origin
            idxs = np.argsort(positions)
            SOPInstanceUIDs = SOPInstanceUIDs[idxs]
            positions = positions[idxs]
            # get class name for local labels
            class_names = []
            for value in tag.iter('value'):
                class_name = value.attrib['name'].split(' (')[0]
                class_names.append(class_name)    
            # get pixel value for local labels
            points = []
            inlabel_z = []
            original_z = []
            for value in point.iter('value'):
                x,y,z = float(value.attrib['x']), float(value.attrib['y']), float(value.attrib['z'])
                original_z.append(z)
                z = (z-np.min(positions))/(np.max(positions)-np.min(positions)) * positions.shape[0]
                points.append([x,y,z])
                inlabel_z.append(z)
            points = np.array(points)
            inlabel_z = sorted(list(set(inlabel_z)))
            points_processes = []
            for ilbz in inlabel_z:
                points_z = points[points[...,2] == ilbz]
                points_processes.append([points_z[:,0].min(), points_z[:,1].min(), points_z[:,0].max(), points_z[:,1].max(), ilbz])
            points_processes = np.array(points_processes)
            inlabel_z_int = list(range(math.ceil(inlabel_z[0]), math.floor(inlabel_z[-1]) + 1))
            points_out = []
            for zint in inlabel_z_int:
                for i in range(len(points_processes) - 1):
                    if points_processes[i,-1] < zint and points_processes[i+1,-1] > zint:
                        portion_1 = zint - points_processes[i,-1]
                        portion_2 = points_processes[i+1,-1] - zint
                        points_out.append([
                            int(((points_processes[i,0] * portion_2 + points_processes[i+1,0] * portion_1) / (portion_1 + portion_2) - xmin) / pixelspacing[0]),
                            int(((points_processes[i,1] * portion_2 + points_processes[i+1,1] * portion_1) / (portion_1 + portion_2) - ymin) / pixelspacing[1]),
                            int(((points_processes[i,2] * portion_2 + points_processes[i+1,2] * portion_1) / (portion_1 + portion_2) - xmin) / pixelspacing[0]),
                            int(((points_processes[i,3] * portion_2 + points_processes[i+1,3] * portion_1) / (portion_1 + portion_2) - ymin) / pixelspacing[1]),
                            SOPInstanceUIDs[int(zint)],
                            zint
                        ])
            points_out = np.array(points_out)
            for class_name in class_names:
                for i in range(len(points_out)):
                    study_out.append([studyUid, seriesUid, points_out[i,4], points_out[i, 5], class_name, int(points_out[i,0]), int(points_out[i,1]), int(points_out[i,2]), int(points_out[i,3])])
    return study_out


if __name__ == "__main__":
    _1511_studies = os.listdir(XML_DIR)
    df = pd.DataFrame()
    for study_i in tqdm(_1511_studies):
        try:
            study_out = prepare_image_process(study_i)
            df_i = pd.DataFrame(data=study_out, columns=["studyUid", "seriesUid", "imageUid", "z-axis-slices", "label", "x1", "x2", "y1", "y2"])
            df = pd.concat([df, df_i])
        except:
            print(study_i)
        
    df.to_csv("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset.csv", index=False)
    df.to_pickle("/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset.pkl")