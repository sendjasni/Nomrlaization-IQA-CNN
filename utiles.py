# Python-Standard librairies
from timeit import default_timer as timer
import random as python_random
import math

# External librairies
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from pathlib import Path

# Internal
from dataset import Dataset

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def create_callbacks_fun(log_dir: Path, weights_dir: Path, fold_no, save_name='tmp', early_stop_bool=True, patience=30, save_weights=False):
    # log_file = log_dir / ('Fold_' + str(fold_no) + '.log')
    log_file = log_dir / (save_name + '.log')

    csv_loger = tf.keras.callbacks.CSVLogger(
        log_file, separator=",", append=False)

    filepath = weights_dir / (save_name + '.hdf5')

    train_tile_cb = TimingCallback()

    callbacks = [csv_loger,
                 train_tile_cb]

    if early_stop_bool:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=patience, restore_best_weights=True)
        callbacks.append(early_stop)

    if save_weights:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", mode="min",
                                                        save_best_only=True, save_weights_only=True, save_freq='epoch',
                                                        verbose=1)
        callbacks.append(checkpoint)

    return callbacks


def get_quick_dataset(inRGB=False) -> np.ndarray:
    """
    Retourne un dataset pour tester rapidement des algorithmes de normalisation
    :return: dataset (imgs=4, w=512, h=512, colors=3)
    """
    if inRGB:
        i1 = np.asarray(Image.open('database/CSIQ/src_imgs/foxy.png'))
        i2 = np.asarray(Image.open('database/CSIQ/src_imgs/aerial_city.png'))
        i3 = np.asarray(Image.open('database/CSIQ/src_imgs/monument.png'))
        i4 = np.asarray(Image.open('database/CSIQ/src_imgs/lake.png'))
        dataset = np.zeros((4, 512, 512, 3))
        dataset[0, :, :, :] = i1
        dataset[1, :, :, :] = i2
        dataset[2, :, :, :] = i3
        dataset[3, :, :, :] = i4
    else:
        i1 = np.asarray(Image.open('database/CSIQ/src_imgs/foxy.png').convert('L'))
        i2 = np.asarray(Image.open('database/CSIQ/src_imgs/aerial_city.png').convert('L'))
        i3 = np.asarray(Image.open('database/CSIQ/src_imgs/monument.png').convert('L'))
        i4 = np.asarray(Image.open('database/CSIQ/src_imgs/lake.png').convert('L'))
        dataset = np.zeros((4, 512, 512, 1))
        dataset[0, :, :, 0] = i1
        dataset[1, :, :, 0] = i2
        dataset[2, :, :, 0] = i3
        dataset[3, :, :, 0] = i4

    return dataset.astype(int)


def apply_seed(seed: int):
    """To fix randomness, use this function with a number
    This will fix a well-defined initial state for each librairies"""
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def plot10Images(images_arr):  # images_arr de 10 images
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_one_img(nparray_img: np.ndarray, i_img: int):
    plt.imshow(nparray_img[i_img, :, :, :].astype(int), interpolation='nearest')


def plot_multiple_img(nparray_img: np.ndarray, use_gray):
    fig = plt.figure(figsize=(30, 30))  # width, height in inches
    plt.gca().axis('off')
    for i in range(15):
        sub = fig.add_subplot(1, 15, i + 1)
        if use_gray:
            sub.imshow(nparray_img[i, :, :, 0].astype(int), interpolation='nearest')
        else:
            sub.imshow(nparray_img[i, :, :, :].astype(int), interpolation='nearest')
    plt.show()


def preprocessing_images(nparray_img: np.ndarray, preprocessing_type: str):
    """s'occupe du pré-traitement d'image avant entrainement"""

    if preprocessing_type == 'SCALE_0_1':  # On parle de scale [0,255]->[0,1] ou que min de chaque img=0 et max=1 (normalisation) ?
        nparray_img = nparray_img / 255.0
    elif preprocessing_type == 'CENTERING':
        perpixel_mean = np.mean(nparray_img, axis=0)  # axis:None = Moyenne sur le tableau après .flatten()
        print("perpixelmeanshape", perpixel_mean.shape)
        nparray_img = nparray_img - perpixel_mean
    elif preprocessing_type == 'STANDARD_SIGMA':
        perpixel_mean = np.mean(nparray_img, axis=None)
        sigma = np.std(nparray_img, axis=None)
        nparray_img = (nparray_img - perpixel_mean) / sigma
    return nparray_img
    # for i in range(nb_images):
    #   img_tmp = nparray_img[i,:,:,:]

    #   nparray_img[i,:,:,:] = img_tmp


def triplicate_gray_channel(dataset_X: np.ndarray) -> np.ndarray:
    """
    >>> dataset = get_quick_dataset(inRGB=False)
    >>> print(dataset.shape)
    (4, 512, 512, 1)
    >>> dataset = triplicate_gray_channel(dataset)
    >>> print(dataset.shape)
    (4, 512, 512, 3)
    """
    # dataset_X = np.stack((dataset_X[:, :, :, 0],)*3, axis=3)
    dataset_X = np.stack((dataset_X.squeeze(3),)*3, axis=3)
    return dataset_X


def save_perf_save_plot(fig, predictions: np.ndarray, test_mos: np.ndarray, dir_path: Path, name: str = 'default'):
    """Sauvegarde les résultats de prédiction d'un test de réseau entraîné"""

    pathname = str(dir_path / name)

    save_perf(predictions, test_mos, dir_path, name)

    fig.savefig(pathname)


def save_perf(predictions: np.ndarray, test_mos: np.ndarray, dir_path: Path, name: str = 'default'):
    """Sauvegarde les résultats de prédiction d'un test de réseau entraîné"""

    concat_test_result = np.array([predictions.flatten(), test_mos.flatten()]).T

    pathname = str(dir_path / name)
    # np.savetxt(pathname + ".csv", concat_test_result, delimiter=",")
    pd.DataFrame(concat_test_result).to_csv(pathname+".csv", header=["p_mos", "mos"], index=False)


def plot_training_history(training_history):
    plt.plot(training_history.history['loss'], label='train')
    plt.plot(training_history.history['val_loss'], label='val')
    plt.legend()
    plt.show()


def plot_test_prediction(test_mos, predictions, name=None, saveplot=False, database_dir=Path("database/")):
    plt.scatter(test_mos, predictions)
    plt.xlabel("MOS")
    plt.ylabel("Predicted MOS")
    fig = plt.gcf()  # Keep the figure even after show() to save it later
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    if name:
        plt.title(name)

    plt.show()

    if saveplot:
        save_perf_save_plot(fig, predictions, test_mos, database_dir)

    return fig