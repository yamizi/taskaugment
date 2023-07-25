import os
import pandas as pd

rootdir = 'D:\datasets\ROBIN-cls-train'
images = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        images.append({"class":subdir.split("\\")[-1], "path":file})

df = pd.DataFrame(images)
df.to_csv("robin_train.csv")