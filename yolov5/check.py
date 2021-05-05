#%%
import pandas as pd
import numpy as np
# from matplotlib.pyplot import hist
from numpy import histogram

import matplotlib.pyplot as plt

#%%

df = pd.read_pickle("/home/single3/Documents/tintrung/yolov5_2classes/yolov5/trung_result/kfold_flair.pkl")
# P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
df.columns = ["Precision","Recall","map0.5","map0.95","5","6","7","8","9"]
histogram(df["map0.5"])



# %%
histogram(df["2"], density=True)
# %%

# df = pd.read_pickle("brain-mri-xml-dataset_2classes.pkl")
# len(df[df.SequenceType =="FLAIR"]["imageUid"])
# %%
fig,a =  plt.subplots(2,2)
# a.grid(True)
fig.set_figheight(15)
fig.set_figwidth(15)
a[0][0].hist(df["Precision"])
mean = np.mean(df["Precision"])
std = np.std(df["Precision"])
a[0][0].set_title(f'Precision (mean: {mean:.3f} std: {std:.3f})')
a[0][0].grid(True)
a[0][1].hist(df["Recall"])
mean = np.mean(df["Recall"])
std = np.std(df["Recall"])
a[0][1].set_title(f'Recall (mean: {mean:.3f} std: {std:.3f})')
a[0][1].grid(True)
a[1][0].hist(df["map0.5"])
mean = np.mean(df["map0.5"])
std = np.std(df["map0.5"])
a[1][0].set_title(f'map@.5 (mean: {mean:.3f} std: {std:.3f})')
a[1][0].grid(True)
a[1][1].hist(df["map0.95"])
mean = np.mean(df["map0.95"])
std = np.std(df["map0.95"])
a[1][1].set_title(f'map@.5-.95 (mean: {mean:.3f} std: {std:.3f})')
a[1][1].grid(True)
plt.show()
plt.tight_layout()
fig.savefig("a.jpg")

# %%
df = pd.read_pickle("/home/single3/Documents/tintrung/yolov5_2classes/yolov5/kfold_t1c.pkl")
# P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
df.columns = ["Precision","Recall","map0.5","map0.95","5","6","7","8","9"]

#%%
fig,a =  plt.subplots(2,2)
# a.grid(True)
fig.set_figheight(15)
fig.set_figwidth(15)
a[0][0].hist(df["Precision"])
mean = np.mean(df["Precision"])
std = np.std(df["Precision"])
a[0][0].set_title(f'Precision (mean: {mean:.3f} std: {std:.3f})')
a[0][0].grid(True)
a[0][1].hist(df["Recall"])
mean = np.mean(df["Recall"])
std = np.std(df["Recall"])
a[0][1].set_title(f'Recall (mean: {mean:.3f} std: {std:.3f})')
a[0][1].grid(True)
a[1][0].hist(df["map0.5"])
mean = np.mean(df["map0.5"])
std = np.std(df["map0.5"])
a[1][0].set_title(f'map@.5 (mean: {mean:.3f} std: {std:.3f})')
a[1][0].grid(True)
a[1][1].hist(df["map0.95"])
mean = np.mean(df["map0.95"])
std = np.std(df["map0.95"])
a[1][1].set_title(f'map@.5-.95 (mean: {mean:.3f} std: {std:.3f})')
a[1][1].grid(True)
plt.show()
plt.tight_layout()
fig.savefig("b.jpg")

# %%
