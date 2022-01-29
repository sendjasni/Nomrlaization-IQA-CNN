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


def run(env):
    """Normalization, no shuffle, and patching images"""

    if env == "COLAB":
        import constants_for_colab as cst
    elif env == "LOCAL":
        import constants as cst

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

    nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv(cst.CSV_DATAFILE_DIR / cst.DATAFILE_NAME,
                                                                         cst.ROOT_DST_IMG, verbose=True, gray=cst.USE_GRAY)

    originals_names = dataframe['image'].unique()
    nb_distor = len(dataframe['dst_type'].unique()) * len(dataframe['dst_lev'].unique())
    nb_images = nparray_label.size

    print(f"Originaux : {originals_names}")
    print(f"Nb distorsions différentes : {nb_distor}")

    # Checking imgs
    plot_multiple_img(nparray_img)

    # Preprocessing images (applying [0, 1] scale to img, centering img values, standardizing, etc.)

    print("Starting patches...")
    patches, mos = Dataset.data_augmentation_patching(nparray_img,
                                                     nb_division=cst.DATA_AUGMENT_FACTOR,
                                                     labels=nparray_label, verbose=True)
    print("Done.")


    print("Starting norm...")
    images = Normalization.preprocess_images(patches, cst.PREPROCESS_IMG)
    print("Done.")

    print(images.shape)
    plt.imshow(images[0,:,:,:])
    plt.show()

    print(f"Saving image np.ndarray{images.shape}...")
    if cst.USE_GRAY:
        np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_GRAY.npy", images)
        np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_GRAY_labels.npy", mos)
    else:
        np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}.npy", images)
        np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_labels.npy", mos)
    print("Done.")


