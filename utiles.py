# Python-Standard librairies
from timeit import default_timer as timer
import random as python_random

# External librairies
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from pathlib import Path

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def create_callbacks_fun(path, fold_no, Normalization):
    log_file = path + '/Fold_' + str(fold_no) + '.log'

    csv_loger = tf.keras.callbacks.CSVLogger(
        log_file, separator=",", append=False)

    filepath = path + '/weights_' + Normalization + '.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", mode="min",
                                                    save_best_only=True, save_weights_only=True, save_freq='epoch',
                                                    verbose=1)
    train_tile_cb = TimingCallback()
    callbacks = [
        csv_loger,
        train_tile_cb,
    ]

    return callbacks


def apply_seed(seed: int):
    """To fix randomness, use this function with a number
    This will fix a well-defined initial state for each librairies"""
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def get_dataset_from_csv(csvfile: str, database_dir: Path):
    """The csvfile must be the name of a .csv file in database_dir/
    Images must be in database_dir/dst_imgs/
    with the name format [photo_name].[distorsion_type].[distorsion_level].png"""

    dataframe = pd.read_csv(database_dir / csvfile)

    nparray_img = np.empty((0, 512, 512, 3))
    nparray_label = np.empty((0, 1, 1, 1))

    for i, row in dataframe.iterrows():
        img_path = f"{str(database_dir / 'dst_imgs' / row['image'])}.{row['dst_type']}.{str(row['dst_lev'])}.png"

        img_PIL = keras.preprocessing.image.load_img(img_path)
        img_np = keras.preprocessing.image.img_to_array(img_PIL)
        # print(f"shape img np : {img_np.shape}")
        # print(f"shape nparray_img : {nparray_img.shape}")

        nparray_img = np.append(nparray_img, img_np[np.newaxis, ...], axis=0)
        quality_score = np.array([row['dmos']])
        nparray_label = np.append(nparray_label, quality_score[np.newaxis, np.newaxis, np.newaxis, ...], axis=0)

    return nparray_img, nparray_label, dataframe


def plot_one_img(nparray_img: np.ndarray, i_img: int):
    plt.imshow(nparray_img[i_img, :, :, :].astype(int), interpolation='nearest')


def plot_multiple_img(nparray_img: np.ndarray):
    fig = plt.figure(figsize=(30, 30))  # width, height in inches
    plt.gca().axis('off')
    for i in range(15):
        sub = fig.add_subplot(1, 15, i + 1)
        sub.imshow(nparray_img[i, :, :, :].astype(int), interpolation='nearest')


def preprocessing_images(nparray_img: np.ndarray, preprocessing_type: str):
    """s'occupe du pré-traitement d'image avant entrainement"""

    if preprocessing_type == 'SCALE_0_1':  # On parle de scale [0,255]->[0,1] ou que min de chaque img=0 et max=1 (normalisation) ?
        nparray_img = nparray_img / 255.0
    elif preprocessing_type == 'CENTERING':
        mean = np.mean(nparray_img, axis=None)  # axis:None = Moyenne sur le tableau après .flatten()
        nparray_img = nparray_img - mean
    elif preprocessing_type == 'STANDARD_SIGMA':
        mean = np.mean(nparray_img, axis=None)
        sigma = np.std(nparray_img, axis=None)
        nparray_img = (nparray_img - mean) / sigma
    return nparray_img
    # for i in range(nb_images):
    #   img_tmp = nparray_img[i,:,:,:]

    #   nparray_img[i,:,:,:] = img_tmp


def convert_permutoriginaux_to_permutlabels(permut_originaux: np.ndarray, nb_distor: int):
    """A partir d'une permutation pour les classes uniques et du nombre de distorsions que subira chacune
  de ces classes, retourne la permutation équivalente pour toutes les images distordues qui gardera le
  même ordre de classes uniques
  >>>permutlabels = convert_permutoriginaux_to_permutlabels(np.array([1, 2, 0]), 3)
  >>>labels = np.array([1,1,1, 2,2,2, 3,3,3])
  >>>print(labels[permutlabels])"""

    permutation_tmp = []
    for elmt in permut_originaux:
        permutation_tmp += list(range(elmt * nb_distor, (elmt + 1) * nb_distor))

    return permutation_tmp


def apply_permut_nparray(nparray: np.ndarray, permut: list):
    """Applique une permutation déjà générée à un tableau numpy"""
    return nparray[permut, :, :, :]


def regression_block(features, initializer):
    x = tf.keras.layers.Dense(1024, activation="relu",
                              kernel_initializer=initializer)(features)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu",
                              kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Dense(
        1, activation='linear', kernel_initializer=initializer)(x)

    return x


def create_model(input_shape, initializer, filters=(32, 64, 128, 256, 512)):
    chan_dim = -1
    inputs = tf.keras.layers.Input(shape=input_shape)

    for (i, f) in enumerate(filters):
        if i == 0:
            x = inputs
        x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
            3, 3), padding="same", kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
            3, 3), padding="same", kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

        max_p = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2))(x)
        avrg_p = tf.keras.layers.AveragePooling2D(
            pool_size=(2, 2), strides=(2, 2))(x)
        # min_p = tf.keras.layers.MaxPooling2D( # pas sur de si je devais ajouter ca
        #     pool_size=(2, 2), strides=(2, 2))(x)

        x = tf.keras.layers.concatenate([avrg_p, max_p])
        # x = tf.keras.layers.concatenate([avrg_p, max_p, min_p])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    output = regression_block(x, initializer)

    iqa_model = tf.keras.models.Model(inputs=inputs, outputs=output)
    return iqa_model


def save_perf_and_plot(fig, predictions: np.ndarray, test_mos: np.ndarray, database_dir: Path, name: str = 'default'):
    concat_test_result = np.array([predictions.flatten(), test_mos.flatten()])

    pathname = str(database_dir / "perfs" / name)
    np.savetxt(pathname + ".csv", concat_test_result, delimiter=",")
    fig.savefig(pathname)
