import numpy as np
import math
import tensorflow as tf
from model import Model
from utiles import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from normalization import Normalization
from numbers import Number


# Parameters #

env = "LOCAL"
DATABASE_NAME = "CSIQ"
PREPROCESS_IMG = "scale_0_1"
USE_GRAY = False

if env == "LOCAL":
    ROOT_PROJECT = Path('./')
elif env == "COLAB":
    ROOT_PROJECT = Path("drive/MyDrive/ICIP_2022/code")
else:
    raise Exception("env must be 'LOCAL' or 'COLAB'")
ROOT_DATABASE = ROOT_PROJECT / "database"


if DATABASE_NAME == "CSIQ":
    path_csv = ROOT_DATABASE / DATABASE_NAME / "CSIQ_DMOS.csv"
elif DATABASE_NAME == "TID2013":
    path_csv = ROOT_DATABASE / DATABASE_NAME / "mos_with_names.csv"
elif DATABASE_NAME == "LIVE2":
    path_csv = ROOT_DATABASE / DATABASE_NAME / "dmos.csv"
else:
    raise Exception('database "'+DATABASE_NAME+'" not found.')
input_shape = (64, 64, 3) if not USE_GRAY else (64, 64, 1)


gray_str = '_GRAY' if USE_GRAY else ''
preprocess_npy_str = 'nothing'
path_dataset = ROOT_DATABASE / DATABASE_NAME / "normalized_dst_imgs" / f"{DATABASE_NAME}_{preprocess_npy_str}_patch{64}{gray_str}"

print(f"Loading database... ({path_dataset}.npy)")
imgs = np.load(f"{path_dataset}.npy")
labels = np.load(f"{path_dataset}_labels.npy")
labels = np.squeeze(labels)
print("Done.")

nb_patches = imgs.shape[0]
indices_images = np.random.randint(nb_patches, size=10)


echantillon_images_before = np.empty((10, input_shape[0], input_shape[1], input_shape[2]))
for mini_i, value in enumerate(indices_images):
    echantillon_images_before[mini_i, :, :, :] = imgs[value, :, :, :]


def plotImages(images_arr):  # images_arr de 10 images
    fig, axes = plt.subplots(1, 10, figsize=(2, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(echantillon_images_before)
print(f"Min/max of dataset before norm : ({np.amin(imgs)}/{np.amax(imgs)})")

print(f"Preprocessing on entire dataset ({PREPROCESS_IMG})...")
imgs = Normalization.preprocess_images(imgs, PREPROCESS_IMG)
print("Done.")


echantillon_images_after = np.empty((10, input_shape[0], input_shape[1], input_shape[2]))
for mini_i, value in enumerate(indices_images):
    echantillon_images_after[mini_i, :, :, :] = imgs[value, :, :, :]
plotImages(echantillon_images_after)
print(f"Min/max of dataset after norm : ({np.amin(imgs)}/{np.amax(imgs)})")
