# Internal files
from utiles import *

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

# Init
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number GPUs avalable: ", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(tf.__version__)

# Constants

# NO_TRAINING = False  # just recover older training
POSSIBLE_PREPROCESSES_IMG = ['SCALE_0_1', 'CENTERING', 'STANDARD_SIGMA']
PREPROCESS_IMG = 'STANDARD_SIGMA'
CSV_SHEET_DATABASE = 'CSIQ_DMOS_allAWGN.csv'
ROOT_DATABASE = Path("./database")  # includes src_imgs/, dst_imgs/ et CSIQ_DMOS.csv
SAVE_TRAINING = True
TEST_BASE_PERCENTAGE = 0.2
apply_seed(seed=111)
PREPROCESS_ALREADY_APPLIED = False

# Image read and imageset

nparray_img, nparray_label, dataframe = get_dataset_from_csv(CSV_SHEET_DATABASE, ROOT_DATABASE)

originaux_names = dataframe['image'].unique()
nb_distor = len(dataframe['dst_type'].unique()) * len(dataframe['dst_lev'].unique())
nb_images = nparray_label.size

print(f"Shape du tensor d'images : {nparray_img.shape}")
print(f"shape des labels correspondants : {nparray_label.shape}")
print(f"Originaux : {originaux_names}")
print(f"Nb distorsions différentes : {nb_distor}")

# Checking imgs

plot_multiple_img(nparray_img)

# Preprocessing images (applying [0, 1] scale to img, centering img values, standardizing, etc.)

nparray_img_processed = preprocessing_images(nparray_img, PREPROCESS_IMG)
print(f"Valeur min des images : {nparray_img_processed.min()}, valeur max des images : {nparray_img_processed.max()}")

# Shuffle and splitting uniques originals (imgs names without distorsions)

permut_originaux, originaux_names_permuted = create_base_permut_and_split(originaux_names)

# Shuffle et séparations des scores associés aux images traitées

permutlabels = convert_permutoriginaux_to_permutlabels(permut_originaux, nb_distor)
nparray_label_permuted = apply_permut_nparray(nparray_label, permutlabels)
labels_test, labels_train = split_test_and_train(nparray_label_permuted, nb_distor, TEST_BASE_PERCENTAGE)

print(f"Permutation originaux : {permut_originaux},\n permut images ({len(permutlabels)}) : {permutlabels}")

print(f"Array to split after same shuffle : \n {nparray_label_permuted.flatten()}")

# Shuffle et sépérations des images traitées

nparray_img_permuted = apply_permut_nparray(nparray_img_processed, permutlabels)

if not PREPROCESS_IMG:
    plot_multiple_img(nparray_img_processed)
    plot_multiple_img(nparray_img_permuted)
    # plot_one_img(nparray_img_processed, 0)

img_test, img_train = split_test_and_train(nparray_img_processed, nb_distor, TEST_BASE_PERCENTAGE)

print(f"Les {img_test.shape[0]} premières images sont gardées pour le test")

initializer = tf.keras.initializers.he_normal()

# Hyper-parameters for the training process

input_shape = (512, 512, 3)  # input shape of the model depending on the shape of input images, so be carefull.
# You need to get this from one of the images. It is preferable to start with a shape of (256, 256, C),
# C is the number of channels (If RGB then C=3, else C=1)
batch_size = 16  # mini-batch size
epochs = 100  # number of epochs.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / epochs)  # optimizer for the model's training

# Creating the CNN model
iqa_model = create_model(input_shape, initializer)

# iqa_model.summary()  # An overview on the model (summary of the network)

train_mos = labels_train
test_mos = labels_test
train_images = img_train
test_images = img_test

print(f"dimensions of scores (train): {train_mos.shape}, nb of values : {train_mos.size}")
print(f"dimensions of images (train): {train_images.shape}, nb of values : {train_images.size}")

# Compile and configures the model for training.
iqa_model.compile(optimizer=optimizer, loss='mse', metrics='mae')

# Configure checkpoints to save training

if SAVE_TRAINING:
    checkpoint_filepath = str(ROOT_DATABASE / f'checkpoint_AWGN.ckpt')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        verbose=1,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    fit_callback = [model_checkpoint_callback]
else:
    fit_callback = None

# Trains the model for a fixed number of epochs (iterations on the used dataset)
training_history = iqa_model.fit(x=train_images, y=train_mos, batch_size=batch_size,
                                 epochs=epochs, validation_split=.2, shuffle=True,
                                 callbacks=fit_callback)

# plot training history

plt.plot(training_history.history['loss'], label='train')
plt.plot(training_history.history['val_loss'], label='val')
plt.legend()
plt.show()

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

plt.scatter(test_mos, predictions)
plt.xlabel("MOS")
plt.ylabel("Predicted MOS")

# name = f"perfs_{epochs}epochs_AWGN_only_standardized"
# plt.title(name)

fig = plt.gcf()  # Keep the figure even after show() to save it later
# plt.show()

# save_perf_and_plot(fig, predictions, test_mos)

