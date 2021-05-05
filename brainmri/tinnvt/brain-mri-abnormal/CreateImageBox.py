import pandas as pd
import cv2 
from tqdm import tqdm

# PKL_FILE = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-dataset-2classes.pkl"
PKL_FILE = "/home/single3/Documents/tintrung/brainmri/tinnvt/brain-mri-abnormal/csv_new/brain-mri-xml-bboxes-copy.pkl"
IMG_DIR = "/home/single3/Documents/tintrung/brain-mri-tumor-images-PNG"
# IMG_BOX_DIR = "/home/single3/Documents/tintrung/brain-mri-tumor-images-bboxes"
IMG_BOX_DIR = "/home/single3/Documents/tintrung/brain-mri-tumor-images-bboxes-copy"

df = pd.read_pickle(PKL_FILE)
def draw_box(image_name):
    img = cv2.imread(f"{IMG_DIR}/{image_name}.png")
    box_idx = df[df["imageUid"] == image_name].iloc[:,4:8].values
    box_name = df[df["imageUid"] == image_name].iloc[:,3].values
    for idx, name in zip(box_idx, box_name):
        img = cv2.rectangle(img, (idx[0], idx[1]), (idx[2], idx[3]), (0,0,255), 2)
        img = cv2.putText(img, name, (idx[0], idx[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(f"{IMG_BOX_DIR}/{image_name}.png", img)

for fn in tqdm(df["imageUid"].unique()):
    draw_box(fn)