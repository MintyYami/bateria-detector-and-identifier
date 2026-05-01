import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import peak_local_max

img_titles = ["640", "488"]

def __getData(dir=None):
    # 0: channel 640, 1: channel 488, 2 Bright Field, 3: folder name
    data = []

    if dir is None:
        dir = os.getcwd()
    dir_data = os.getcwd()[:dir.rfind('/')] + "/data"
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

# show images
def printImg(imgs, map=None):
    fig = plt.figure(figsize=(8,15))
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(img_titles[i])
        plt.imshow(imgs[i], cmap=map)
        plt.axis('off')

# show masks
def printMask(masks, map=None):
    fig = plt.figure(figsize=(8,15))
    for i in range(2):
        plt.subplot(1, 3, i+1)
        plt.title(img_titles[i])
        plt.imshow(masks[i][0], cmap=map)
        plt.axis('off')

# show regions with labels
def __show_region(ax, image, stat, map=None):
    ax.imshow(image, cmap=map)
    ax.scatter(
        stat["x"],
        stat["y"],
        s=10,
        facecolors="none",
        edgecolors="red"
    )
    for _, row in stat.iterrows():
        ax.text(
            row["x"],
            row["y"],
            str(int(row["label"])),
            color="yellow",
            fontsize=6
        )
    ax.axis("off")
def printRegion(imgs, stats, map=None):
    fig, axes = plt.subplots(1, 2, figsize=(5,15))
    for i in range(2):
        # add labels
        __show_region(axes[i], imgs[i], stats[i], map=map)
        axes[i].set_title(img_titles[i])
    plt.show()

# show labelled comparisons of regions
def __show_label(ax, image, compared, map=None, legend=False):
    shared, A_only, B_only = compared["shared"], compared["only_640"], compared["only_488"]
    ax.imshow(image, cmap=map)
    # shared
    ax.scatter(
        shared["x"],
        shared["y"],
        s=30,
        facecolors="none",
        edgecolors="lime",
        linewidths=1.5,
        label="shared"
    )
    # only 640
    ax.scatter(
        A_only["x"],
        A_only["y"],
        s=20,
        facecolors="none",
        edgecolors="red",
        linewidths=1,
        label="640 only"
    )
    # only 488
    ax.scatter(
        B_only["x"],
        B_only["y"],
        s=20,
        facecolors="none",
        edgecolors="cyan",
        linewidths=1,
        label="488 only"
    )

    if legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.05, 1),  # push outside to the right
            borderaxespad=0.
        )
    ax.axis("off")
def printLabel(imgs, compared, map=None):
    fig, axes = plt.subplots(1, 2, figsize=(5, 15))
    for i in range(2):
        # add labels
        __show_label(axes[i], imgs[i], compared, map=map,legend=(i == 1))
        axes[i].set_title(img_titles[i])
    plt.show()