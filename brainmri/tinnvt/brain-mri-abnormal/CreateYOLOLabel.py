from tqdm import tqdm
import cv2
import pandas as pd
import shutil
import json

CSV_FILE = "brain-mri-dataset.csv"
IMG_DIR = "/media/datnt/data/medical-image-data/brain-mri-images/"
CLASS_INDEX = {
 'Ventricular dilation': 0,
 'Hemorrhagic component': 1,
 'Other lesion': 2,
 'Subdural hematoma': 3,
 'Demyelination': 4,
 'Mass/Nodule': 5,
 'Cyst component': 6,
 'Sinus lesion': 7,
 'Infiltration': 8,
 'Cerebral edema': 9,
 'Bone lesion': 10,
 'CSF-like lesion': 11,
 'Ischemia': 12,
 'Midline shift': 13,
 'Gliosis': 14,
 'Abscess': 15,
 'Subdural effusion': 16,
 'Cerebral atrophy': 17,
 'Intraparenchymal hemorrhage': 18,
 'Mass effect': 19,
 'Cavernoma': 20,
 'Cerebral vascular disease': 21,
 'Intracranial herniation': 22,
 'Subarachnoid hemorrhage': 23,
 'Calcification': 24,
 'Arteriovenous malformation': 25,
 'Epidural hematoma': 26,
 'Meningitis': 27,
 'Intraventricular hemorrhage': 28,
 'Cerebral contusion': 29}


df = pd.read_csv(CSV_FILE)
counter = 0
translator = {}
for image in tqdm(df["imageUid"].unique()):
# for image in ["1.3.12.2.1107.5.2.40.39308.2020021909502243827163681"]:
    data = []
    sub_df = df[df["imageUid"] == image]    
    img = cv2.imread(IMG_DIR + image + ".jpg")
    h, w, _ = img.shape
    labels = sub_df.iloc[:,3].values
    idx = sub_df.iloc[:,4:8].values
    for lb, ix in zip(labels, idx):
        x = ((ix[2] + ix[0]) / 2) / w
        y = ((ix[3] + ix[1]) / 2) / h
        bw = (ix[2] - ix[0]) / w
        bh = (ix[3] - ix[1]) / h
        data.append([CLASS_INDEX[lb], x, y, bw, bh])
    data = pd.DataFrame(data=data)
    shutil.copy2(IMG_DIR + image + ".jpg", f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/train/{counter}.jpg")
    data.to_csv(f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/train/{counter}.txt", index=False, header=False, sep=" ")    
    translator[image] = counter
    counter += 1

with open('translator.json', 'w') as fp:
    json.dump(translator, fp)

for set_ in ["valid"]:
    set1 = set(pd.read_csv(f"/home/datnt/Code/brain-mri-classification/csv/brain-mri-abnormalness-{set_}-v2.csv")["imageUid"].values)
    set2 = set(pd.read_csv(CSV_FILE)["imageUid"].values)
    fnames = set1.intersection(set2)
    for f in fnames:
        f = translator[f]
        shutil.move(f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/train/{f}.jpg", f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/valid/")
        shutil.move(f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/train/{f}.txt", f"/home/datnt/Code/brain-mri-detection/datasets/brain-mri-single-slice/valid/")