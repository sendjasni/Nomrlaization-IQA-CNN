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


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


def create_callbacks_fun(log_dir: Path, weights_dir: Path, fold_no, Normalization):
    log_file = log_dir / ('Fold_' + str(fold_no) + '.log')

    csv_loger = tf.keras.callbacks.CSVLogger(
        log_file, separator=",", append=False)

    filepath = weights_dir / ('weights_' + Normalization + '.hdf5')

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", mode="min",
                                                    save_best_only=True, save_weights_only=True, save_freq='epoch',
                                                    verbose=1)
    train_tile_cb = TimingCallback()
    callbacks = [
        csv_loger,
        train_tile_cb,
        checkpoint
    ]

    return callbacks


def get_quick_dataset(inRGB=False) -> np.ndarray:
    """
    Retourne un dataset pour tester rapidement des algorithmes de normalisation
    :return: dataset (imgs=4, w=512, h=512, colors=3)
    """
    if inRGB:
        i1 = np.asarray(Image.open('database/src_imgs/foxy.png'))
        i2 = np.asarray(Image.open('database/src_imgs/aerial_city.png'))
        i3 = np.asarray(Image.open('database/src_imgs/monument.png'))
        i4 = np.asarray(Image.open('database/src_imgs/lake.png'))
        dataset = np.zeros((4, 512, 512, 3))
        dataset[0, :, :, :] = i1
        dataset[1, :, :, :] = i2
        dataset[2, :, :, :] = i3
        dataset[3, :, :, :] = i4
    else:
        i1 = np.asarray(Image.open('database/src_imgs/foxy.png').convert('L'))
        i2 = np.asarray(Image.open('database/src_imgs/aerial_city.png').convert('L'))
        i3 = np.asarray(Image.open('database/src_imgs/monument.png').convert('L'))
        i4 = np.asarray(Image.open('database/src_imgs/lake.png').convert('L'))
        dataset = np.zeros((4, 512, 512, 1))
        dataset[0, :, :, 0] = i1
        dataset[1, :, :, 0] = i2
        dataset[2, :, :, 0] = i3
        dataset[3, :, :, 0] = i4

    return dataset


def apply_seed(seed: int):
    """To fix randomness, use this function with a number
    This will fix a well-defined initial state for each librairies"""
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)


def get_dataset_from_csv(csvfilepath: str, database_dir: Path, verbose=False):
    """Images must be in database_dir/dst_imgs/
    with the name format [photo_name].[distorsion_type].[distorsion_level].png"""

    dataframe = pd.read_csv(csvfilepath)

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

    if verbose:
        print(f"Shape du tensor d'images : {nparray_img.shape}")
        print(f"shape des labels correspondants : {nparray_label.shape}")

    return nparray_img, nparray_label, dataframe


def plot_one_img(nparray_img: np.ndarray, i_img: int):
    plt.imshow(nparray_img[i_img, :, :, :].astype(int), interpolation='nearest')


def plot_multiple_img(nparray_img: np.ndarray):
    fig = plt.figure(figsize=(30, 30))  # width, height in inches
    plt.gca().axis('off')
    for i in range(15):
        sub = fig.add_subplot(1, 15, i + 1)
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


def convert_permutoriginals_to_permutlabels(permut_originaux: np.ndarray, nb_distor: int):
    """A partir d'une permutation pour les classes uniques et du nombre de distorsions que subira chacune
  de ces classes, retourne la permutation équivalente pour toutes les images distordues qui gardera le
  même ordre de classes uniques
  >>> permutlabels = convert_permutoriginals_to_permutlabels(np.array([1, 2, 0]), 3)
  >>> labels = np.array([1,1,1, 2,2,2, 3,3,3])
  >>> assert((labels[permutlabels] == np.array([2,2,2, 3,3,3, 1,1,1])).all())"""

    permutation_tmp = []
    for elmt in permut_originaux:
        permutation_tmp += list(range(elmt * nb_distor, (elmt + 1) * nb_distor))

    return permutation_tmp


def apply_permut_nparray(nparray: np.ndarray, permut: list):
    """Applique une permutation déjà générée à un tableau numpy sous la forme [elmt_de_train, width, height,
    canal_couleurs] """
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


def create_model(input_shape, initializer, filters=(64, 128, 256, 512)):
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


def save_perf_save_plot(fig, predictions: np.ndarray, test_mos: np.ndarray, database_dir: Path, name: str = 'default'):
    """Sauvegarde les résultats de prédiction d'un test de réseau entraîné"""

    concat_test_result = np.array([predictions.flatten(), test_mos.flatten()])

    pathname = str(database_dir / "perfs" / name)
    np.savetxt(pathname + ".csv", concat_test_result, delimiter=",")
    fig.savefig(pathname)


def create_base_permut_and_split(originals: np.ndarray, test_percent: int = 0.2, verbose=False):
    """Originals are the names of the src_imgs in a list"""

    permut = np.random.permutation(len(originals))
    originals_permuted = originals[permut]
    split_point = math.ceil(len(originals_permuted) * test_percent)
    originals_test, originals_train = np.split(originals_permuted, [split_point])

    if verbose:
        print(f"Liste des originaux avant shuffle : \n {originals}")
        print(f"Liste des originaux après shuffle : \n {originals_permuted}")
        print(f"Liste des originaux après split point de 20 % min en test : \n {originals_test} ; {originals_train}")

    return permut, originals_permuted


def split_test_and_train(nparray_to_split: np.ndarray, nb_distorsions: int, test_percent: float = 0.2, verbose=False):
    """Splits a ndarray according to the number of distorsions applied to each img"""
    nb_originals_src = nparray_to_split.shape[0] / nb_distorsions
    split_point = math.round(nb_originals_src * test_percent)
    arr_test, arr_train = np.split(nparray_to_split, [split_point * nb_distorsions], axis=0)

    if verbose:
        print(f"Array to split ({nparray_to_split.size}) : \n {nparray_to_split.flatten()}")
        print(f"Array to split after taking {test_percent * 100} % as Test : ")
        print(f"Test ({arr_test.shape[0]}/{nparray_to_split.shape[0]}) : {arr_test.flatten()} ; ")
        print(f"Train ({arr_train.shape[0]}/{nparray_to_split.shape[0]}) : {arr_train.flatten()}")

    return arr_test, arr_train


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

    if name:
        plt.title(name)

    plt.show()
    if saveplot:
        save_perf_save_plot(fig, predictions, test_mos, database_dir)
