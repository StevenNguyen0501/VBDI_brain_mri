#%%
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
#%%
os.chdir("/home/single2/tintrung/VBDI_brain_mri/brainmri/trungdc/")
data = pd.read_pickle("semisupervised/coco_bbox.pkl")
data = data[data.sequence == "flair"]

X = data[['id', 'file_name', 'height', 'width', 'date_captured', 'bbox', 'sequence']]
y = data[ 'category_id']
X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.3, random_state=42)


traindata = X_train.copy()
traindata["category_id"] = y_train 

valdata = X_val.copy()
valdata["category_id"] = y_val


valdata.head()
print(traindata.shape)
print(valdata.shape)


#%%
imageval = valdata[["id", "width", "height", "file_name", "date_captured"]]
annotationval = valdata[["id", "category_id", "bbox"]]
annotationval.columns = ["imageid", "category_id", "bbox"]

imageval.to_json("semisupervised/imageval.json", orient = "records")
annotationval.to_json("semisupervised/annotationval.json", orient = "records")

with open("semisupervised/headerCocoJson.json") as json_file:
    header = json.load(json_file)
with open("semisupervised/annotationval.json") as json_file:
    annotationval = json.load(json_file)
with open("semisupervised/imageval.json") as json_file:
    imageval = json.load(json_file)
header["images"] = imageval
header["annotations"] = annotationval

with open('semisupervised/FLAIR_COCO_val.json', 'w') as outfile:
    json.dump(header, outfile)



# %%
imagetrain = traindata[["id", "width", "height", "file_name", "date_captured"]]
annotationtrain = traindata[["id", "category_id", "bbox"]]
annotationtrain.columns = ["imageid", "category_id", "bbox"]

imagetrain.to_json("semisupervised/imagetrain.json", orient = "records")
annotationtrain.to_json("semisupervised/annotationtrain.json", orient = "records")

with open("semisupervised/headerCocoJson.json") as json_file:
    header = json.load(json_file)
with open("semisupervised/annotationtrain.json") as json_file:
    annotationtrain = json.load(json_file)
with open("semisupervised/imagetrain.json") as json_file:
    imagetrain = json.load(json_file)
header["images"] = imagetrain
header["annotations"] = annotationtrain

with open('semisupervised/FLAIR_COCO_train.json', 'w') as outfile:
    json.dump(header, outfile)

# %%
traindata.to_pickle("semisupervised/FLAIR_train_COCO.pkl")
valdata.to_pickle("semisupervised/FLAIR_val_COCO.pkl")
# %%
