import pandas as pd
import cv2 
from tqdm import tqdm


PKL_FILE = "./csv_new/brain-mri-xml-bboxes-copy-final.pkl"
IMG_DIR = "/home/single3/tintrung/brain-mri-tumor-images-PNG"
IMG_BOX_DIR = "/home/single3/tintrung/brain-mri-tumor-images-bboxes-copy-final"

df = pd.read_pickle(PKL_FILE)
IMG_DIR = "/home/single3/tintrung/images"
IMG_BOX_DIR = "/home/single3/tintrung/brain-mri-tumor-images-bboxes-final"

def draw_bboxes_compare_two_sequences(image_based_name, image_copied_name):
    img1 = cv2.imread(f"{IMG_DIR}/{image_copied_name}.png")
    box_idx1 = df[df["copied_imageUid"] == image_copied_name].iloc[:,11:15].values
    box_name1 = df[df["copied_imageUid"] == image_copied_name].iloc[:,1].values
    img2 = cv2.imread(f"{IMG_DIR}/{image_based_name}.png")
    box_idx2 = df[df["based_imageUid"] == image_based_name].iloc[:,5:9].values
    box_name2 = df[df["based_imageUid"] == image_based_name].iloc[:,1].values
    
    for idx1, name1 in zip(box_idx1, box_name1):
        img1 = cv2.rectangle(img1, (int(idx1[0]), int(idx1[1])), (int(idx1[2]), int(idx1[3])), (0,0,255), 2)
        img1 = cv2.putText(img1, name1, (int(idx1[0]), int(idx1[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    for idx2, name2 in zip(box_idx2, box_name2):
        img2 = cv2.rectangle(img2, (int(idx2[0]), int(idx2[1])), (int(idx2[2]), int(idx2[3])), (0,0,255), 2)
        img2 = cv2.putText(img2, name2, (int(idx2[0]), int(idx2[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
    
    img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
    img = cv2.hconcat([img1, img2])

    cv2.imwrite(f"{IMG_BOX_DIR}/{image_copied_name}_{image_based_name}.png", img)

for index, row in df.iterrows():
    try:   
        draw_bboxes_compare_two_sequences(row['based_imageUid'], row['copied_imageUid'])
    except Exception as e:
        print(e)