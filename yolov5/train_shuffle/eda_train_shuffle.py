#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
#%%
df = pd.read_pickle("/home/single3/Documents/tintrung/yolov5_2classes/yolov5/train_shuffle/saved_result/kfold_combine_three.pkl")
df.columns = ["Precision","Recall","map0.5","map0.95","5","6","7","8","9"]
df.head()
# %%
save_im_path = "/home/single3/Documents/tintrung/yolov5_2classes/yolov5/train_shuffle/saved_image"
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
plt.title("2fold 50 resample yolov5s with three sequence")
fig.savefig(os.path.join(save_im_path,"three_combine.jpg"))

# %%

df = pd.read_pickle("/home/single3/Documents/tintrung/yolov5_2classes/yolov5/train_shuffle/saved_result/kfold_flair.pkl")
# P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
df.columns = ["Precision","Recall","map0.5","map0.95","5","6","7","8","9"]

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
plt.title("2fold 50 resample yolov5s with FLAIR sequence")
fig.savefig(os.path.join(save_im_path,"flair.jpg"))

# %%
df = pd.read_pickle("/home/single3/Documents/tintrung/yolov5_2classes/yolov5/train_shuffle/saved_result/kfold_t1c.pkl")
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
fig.subtitle("2fold 50 resample yolov5s with T1C sequence")
plt.show()
plt.tight_layout()

fig.savefig(os.path.join(save_im_path,"t1c.jpg"))
# %%

# %%

# %%

# %%

# %%
