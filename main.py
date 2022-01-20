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
                                                                         cst.ROOT_DATABASE, verbose=True)

    originals_names = dataframe['image'].unique()
    nb_distor = len(dataframe['dst_type'].unique()) * len(dataframe['dst_lev'].unique())
    nb_images = nparray_label.size

    print(f"Originaux : {originals_names}")
    print(f"Nb distorsions différentes : {nb_distor}")

    # Checking imgs
    plot_multiple_img(nparray_img)

    # Preprocessing images (applying [0, 1] scale to img, centering img values, standardizing, etc.)

    nparray_img_processed = Normalization.preprocess_images(nparray_img, cst.PREPROCESS_IMG)

    print(
        f"Valeur min des images : {nparray_img_processed.min()}, valeur max des images : {nparray_img_processed.max()}")

    # Shuffle and splitting uniques originals (imgs names without distorsions)

    permut_originals, originals_names_permuted = Dataset.create_base_permut_and_split(originals_names)

    # Shuffle et séparations des scores associés aux images traitées
    # Shuffle et sépérations des images d'entrainement

    permutlabels = Dataset.convert_permutoriginals_to_permutlabels(permut_originals, nb_distor)

    nparray_label_permuted = Dataset.apply_permut_nparray(nparray_label, permutlabels)
    nparray_img_permuted = Dataset.apply_permut_nparray(nparray_img_processed, permutlabels)

    test_mos_before_augment, train_mos_before_augment = Dataset.split_test_and_train(nparray_label_permuted, nb_distor,
                                                                                     cst.TEST_BASE_PERCENTAGE)
    test_images_before_augment, train_images_before_augment = Dataset.split_test_and_train(nparray_img_processed,
                                                                                           nb_distor,
                                                                                           cst.TEST_BASE_PERCENTAGE)

    print(
        f"Before data_augmentation : trainshape {train_images_before_augment.shape};{train_mos_before_augment.shape}; testshape {test_images_before_augment.shape};{test_mos_before_augment.shape}")
    train_images, train_mos = Dataset.data_augmentation_patching(train_images_before_augment,
                                                                 nb_division=cst.DATA_AUGMENT_FACTOR,
                                                                 labels=train_mos_before_augment)
    test_images, test_mos = Dataset.data_augmentation_patching(test_images_before_augment,
                                                               nb_division=cst.DATA_AUGMENT_FACTOR,
                                                               labels=test_mos_before_augment)
    print(
        f"After data_augmentation : trainshape {train_images.shape};{train_mos.shape}; testshape {test_images.shape};{test_mos.shape}")

    print(f"Permutation originals : {permut_originals},\n permut images ({len(permutlabels)}) : {permutlabels}")
    print(f"Array to split after same shuffle : \n {nparray_label_permuted.flatten()}")
    print(f"Shape : \n {nparray_img_permuted.shape}")

    if not cst.PREPROCESS_IMG:
        plot_multiple_img(nparray_img_processed)
        plot_multiple_img(nparray_img_permuted)
        # plot_one_img(nparray_img_processed, 0)

    print(f"Les {test_images.shape[0]} premières images sont gardées pour le test")

    initializer = tf.keras.initializers.he_normal()

    # Creating the CNN model
    iqa_model = Model.create_model(cst.input_shape, initializer)

    # iqa_model.summary()  # An overview on the model (summary of the network)

    print(f"dimensions of scores (train): {train_mos.shape}, nb of values : {train_mos.size}")
    print(f"dimensions of images (train): {train_images.shape}, nb of values : {train_images.size}")

    # Compile and configures the model for training.
    iqa_model.compile(optimizer=cst.optimizer, loss='mse', metrics='mae')

    # Configure checkpoints to save training

    if cst.SAVE_TRAINING:
        callbacks_arr = create_callbacks_fun(cst.ROOT_DATABASE, cst.SAVED_TRAINING_DIR, fold_no=0, Normalization=cst.PREPROCESS_IMG)
        fit_callbacks = callbacks_arr
    else:
        fit_callbacks = None

    # Trains the model for a fixed number of epochs (iterations on the used dataset)

    print("Checkpoint 1")
    training_history = iqa_model.fit(x=train_images, y=train_mos, batch_size=cst.batch_size,
                                     epochs=cst.epochs, validation_split=cst.VALID_BASE_PERCENTAGE, shuffle=True,
                                     callbacks=fit_callbacks)
    print("Checkpoint 2")

    # plot loss and val_loss along the training

    plot_training_history(training_history)

    print('[INFO] Evaluating/Predicting...')

    evaluation_score = iqa_model.evaluate(test_images, test_mos, batch_size=8)

    predictions = iqa_model.predict(test_images, batch_size=32)

    print(f'loss = {evaluation_score[0]},  accuracy = {evaluation_score[1]}')
    print(f" Prédictions de scores : {predictions.flatten()}")
    print(f" Vrais scores : {test_mos.flatten()}")
    plcc = pearsonr(predictions.flatten(), test_mos.flatten())
    srocc = spearmanr(predictions.flatten(), test_mos.flatten())
    print(f'PLCC = {plcc[0]}, SROCC = {srocc[0]}')

    # Scatter plot to vosualize the relation between the actual MOS and the
    # predicted one

    name = f"perfs_{cst.epochs}epochs_AWGN_only"
    plot_test_prediction(test_mos, predictions, name)
