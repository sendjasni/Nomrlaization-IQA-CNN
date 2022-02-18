# Internal files
from utiles import *
from normalization import Normalization
from dataset import Dataset
from model import Model

# Python-Standard librairies
import os
import random as python_random
import math

# External librairies
import tensorflow as tf
import numpy as np
from pathlib import Path
import pandas as pd
import keras
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


def run(env, cst=None):
    """Normalization, no shuffle, and patching images"""

    # Parameters #
    if env == "LOCAL":
        import constants as cst
    elif not cst:
        raise ImportError

    # Init

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Number GPUs avalable: ", len(physical_devices))
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    print(tf.__version__)

    if cst.FIX_RANDOM:
        apply_seed(seed=111)  # Controls and fix the randomness of numpy/python/tensorflow (shuffling, etc.)
    # Image read and imageset

    nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv(cst.DATABASE_NAME, cst.ROOT_DATABASE, verbose=True, gray=cst.USE_GRAY)

    # originals_names = dataframe['image'].unique()

    nb_images = nparray_label.size

    # print(f"Originaux : {originals_names}")

    # Checking imgs
    plot_multiple_img(nparray_img, cst.USE_GRAY)

    # Preprocessing images (applying [0, 1] scale to img, centering img values, standardizing, etc.)
    if cst.USE_GRAY:
        plt.imshow(nparray_img[0, :, :, 0]); plt.show()
    else:
        plt.imshow(nparray_img[0, :, :, :]); plt.show()

    print("Starting patches...")
    patches, mos = Dataset.data_augmentation_patching(nparray_img,
                                                     nb_division=cst.DATA_AUGMENT_FACTOR,
                                                     labels=nparray_label, verbose=True)
    print("Done.")

    print(patches.shape)
    if cst.USE_GRAY:
        plt.imshow(patches[0, :, :, 0]); plt.show()
    else:
        plt.imshow(patches[0, :, :, :]); plt.show()

    print("Starting norm...")
    images = Normalization.preprocess_images(patches, cst.PREPROCESS_IMG)
    print("Done.")

    print(images.shape)
    if cst.USE_GRAY:
        plt.imshow(patches[0, :, :, 0]); plt.show()
    else:
        plt.imshow(patches[0, :, :, :]); plt.show()

    print(f"Saving image np.ndarray{images.shape}...")

    gray_str = '_GRAY' if cst.USE_GRAY else ''

    np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{cst.PREPROCESS_IMG}_patch{64}{gray_str}.npy", images)
    np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{cst.PREPROCESS_IMG}_patch{64}{gray_str}_labels.npy", mos)

    print("Done.")


