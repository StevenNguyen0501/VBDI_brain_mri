import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import glob

XML_DIR = ""
OUTPUT = ""

# Read label.xml file
def parse_xml(label_file):
    label_file = label_file
    study_id = label_file.split('/')[-2]
    # Loading label file
    try:
        tree = ET.parse(label_file)
    except FileNotFoundError:
        study_id = label_file.split('/')[-2]
        print(f'Study_id = {study_id} Has no xml file ')
        return None
    
    data = {'studyUid': [],
            'seriesUid': [],
            'imageUid': [],
            'createTimestamp': [],
            'sessionId': [],
            'type': [],
            'annotation': [],
            'name': [],
            'x_pos': [],
            'y_pos': [],
            'z_pos': []}
        
    root = tree.getroot()
    
    global_label = ''
    seriesID = ''
    studyID = ''
    patientpid = ''
    session_id = ''
    
    for p in root.iter('patient'):
        patientpid = p.attrib['pid']
    
    for i, label in enumerate(root.iter('label')):
        if not session_id:
            session_id = label.attrib['sessionId']
        studyID = label.attrib['studyUid']
        if not seriesID:
            seriesID = label.attrib['seriesUid']
        if label.attrib['type'] == 'global':
            for value in label.iter('value'):
                try:
                    if not value.attrib["name"] in global_label:
                        global_label += f'{value.attrib["name"]},'
                except KeyError:
                    continue
                    
        
        for key in data.keys():
            point = label.find('point')
            for j, value in enumerate(point.iter('value')):
                x = float(value.attrib['x'])
                y = float(value.attrib['y'])
                z = float(value.attrib['z'])
                
                if key == 'name':
                    tag = label.find('tags')
                    label_name = ''
                    for value in tag.iter('value'):
                        label_name += f'{value.attrib["name"]}, '
                    data[key].append(label_name)

                elif key == 'x_pos' or key == 'y_pos' or key == 'z_pos':
                    axis = key.split('_')[0]
                    data[key].append(float(value.attrib[axis]))
                else:
                    data[key].append(label.attrib[key])

    df = pd.DataFrame(data)
    global_labels = global_label.split(',')[:-1]
    accepted_timeStamp = []
    for ts in df["createTimestamp"].unique():
        df_ts = df[df["createTimestamp"] == ts]
        if len(df_ts["z_pos"].unique() == 1):
            accepted_timeStamp.append(ts)
    df = df[df["createTimestamp"].isin(accepted_timeStamp)]
    return df, global_labels, studyID, seriesID, patientpid, session_id

xml_list = glob.glob(XML_DIR)
data = []
for xml in tqdm(xml_list):
    study_df, global_labels, studyID, seriesID, patientpid, session_id = parse_xml(xml)
    data.append([session_id, global_labels, studyID, patientpid])
df = pd.DataFrame(data=data, columns=["sessionID", "globalLabels", "studyUID", "PID"])
df["reviewStatus"] = 0
df.to_csv(OUTPUT, index=False)