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

    if cst.DATABASE_NAME != 'LIVE2':
        raise Exception

    if cst.FIX_RANDOM:
        apply_seed(seed=111)  # Controls and fix the randomness of numpy/python/tensorflow (shuffling, etc.)

    all_patches, all_labels, df = Dataset.get_LIVEdataset_from_csv(cst.ROOT_DATABASE / cst.DATABASE_NAME, verbose=True, gray=True)

    plt.imshow(all_patches[96+0, :, :, :], cmap='gray'); plt.show()
    plt.imshow(all_patches[96+1, :, :, :], cmap='gray'); plt.show()
    plt.imshow(all_patches[96+2, :, :, :], cmap='gray'); plt.show()
    plt.imshow(all_patches[96+3, :, :, :], cmap='gray'); plt.show()
    plt.imshow(all_patches[96+4, :, :, :], cmap='gray'); plt.show()

    gray_str = "_GRAY" if cst.USE_GRAY else ""
    np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{cst.PREPROCESS_IMG}_patch{64}{gray_str}.npy", all_patches)
    np.save(cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{cst.PREPROCESS_IMG}_patch{64}{gray_str}_labels.npy", all_labels)
