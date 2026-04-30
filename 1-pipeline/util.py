import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.feature import peak_local_max
# 0: channel 640, 1: channel 488, 2 Bright Field, 3: folder name
data = []

def __getData(dir=None):
    if dir is None:
        dir = os.getcwd()
    dir_data = os.getcwd()[:dir.rfind('\\')] + "\\data"
    for folder in os.listdir(dir_data):
        new_img = [None, None, None, folder]
        if str(folder).endswith(".DS_Store"):
            continue
        for file in os.listdir(os.path.join(dir_data, folder)):
            if str(file).endswith("colour0.tif"):
                tempImg = mpimg.imread(os.path.join(dir_data, folder, file))
                # new_img[0] = np.array(tempImg)
                new_img[0] = np.array([i[tempImg.shape[1]//2:tempImg.shape[1]] for i in tempImg])
            elif str(file).endswith("colour1.tif"):
                tempImg = mpimg.imread(os.path.join(dir_data, folder, file))
                # new_img[1] = np.array(tempImg)
                new_img[1] = np.array([i[:tempImg.shape[1]//2] for i in tempImg])
            elif str(file).endswith("colour2.tif"):
                tempImg = mpimg.imread(os.path.join(dir_data, folder, file))
                # new_img[2] = np.array(tempImg)
                new_img[2] = np.array([i[:tempImg.shape[1]//2] for i in tempImg])
        data.append(new_img)
    data.sort(key = lambda x : int(x[3].split("_")[1]))

    return data

def getAliveData(dir=None):
    data = __getData(dir)
    data_alive = []
    for i in (list(range(43,59)) + list(range(84,102)) + list(range(127, 227))):
        data_alive.append(data[i])
    return data_alive

# image read function
def printImg(imgs, map=None):
    fig = plt.figure(figsize=(8,15))
    fig.suptitle("Image ")
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.imshow(imgs[i], cmap=map)
        plt.axis('off')

# image read function
def printMask(masks, map=None):
    fig = plt.figure(figsize=(8,15))
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.imshow(masks[i][0], cmap=map)
        plt.axis('off')