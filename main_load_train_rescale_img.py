import numpy as np
import math
import tensorflow as tf
from model import Model
from utiles import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from normalization import Normalization
from numbers import Number
from dataset import Dataset
import cv2


def run(env, cst):
    """This codes comes after patching. The data is already patched and saved in .npy files
    if this is run from local computer, don't precise cst argument
    """

    print("Database name : ", cst.DATABASE_NAME)
    if cst.DATABASE_NAME == "CSIQ":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "CSIQ_DMOS.csv"
    elif cst.DATABASE_NAME == "TID2013":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "mos_with_names.csv"
    elif cst.DATABASE_NAME == "LIVE2":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "dmos.csv"
    else:
        raise Exception('database "'+cst.DATABASE_NAME+'" not found.')

    nparray_img, nparray_label, dataframe = Dataset.get_dataset_from_csv(cst.DATABASE_NAME, cst.ROOT_DATABASE, verbose=True, gray=cst.USE_GRAY)

    new_h = nparray_img.shape[1]//3
    new_w = nparray_img.shape[2]//4
    redim_nparray_img = np.empty((nparray_img.shape[0], new_h, new_w, nparray_img.shape[3]))
    for i in range(nparray_img.shape[0]):
        redim_nparray_img[i, :, :, :] = cv2.resize(nparray_img[i, :, :, :], (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    imgs = redim_nparray_img
    labels = np.squeeze(nparray_label)

    NB_PATCH_PER_IMG = cst.DATA_AUGMENT_FACTOR[0] * cst.DATA_AUGMENT_FACTOR[1]

    # for iter in np.arange(0, 1, cst.TEST_BASE_PERCENTAGE)
    # testing_slice = [0, 0.2]

    # for iter in cst.ITERATION:
    # saving paths

    if cst.ITERATION == 'all':
        all_testing_slice = np.append(np.arange(0, 1, cst.TEST_BASE_PERCENTAGE), 1)
    else:
        all_testing_slice = np.append(np.arange(0, 1, cst.TEST_BASE_PERCENTAGE), 1)[cst.ITERATION: cst.ITERATION + 2]

    # all_testing_slice = cst.ALL_TESTING_SLICE

    print("all_testing_slice : ", all_testing_slice)
    print("for : ", range(len(all_testing_slice) - 1))
    print("first testing slice : ", all_testing_slice[0:0+2])

    gray_str = '_GRAY' if cst.USE_GRAY else ''

    for i_iteration in range(len(all_testing_slice) - 1):

        testing_slice = all_testing_slice[i_iteration:i_iteration + 2]
        if isinstance(cst.ITERATION, Number):
            i_iteration = cst.ITERATION

        # i_iteration = i_slice + (round(testing_slice[0] / cst.TEST_BASE_PERCENTAGE))
        print(f"COMMENCEMENT ITERATION {i_iteration} ({testing_slice})")

        name_log_weight = f"iteration{i_iteration}"
        dir_predictions_arr = cst.ROOT_PROJECT / 'test_results' / cst.DATABASE_NAME / cst.PREPROCESS_IMG
        name_predictions_arr = f"{cst.epochs}epochs_iteration{i_iteration}{gray_str}"
        dir_logs = cst.LOGS_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG
        dir_weights = cst.SAVED_TRAINING_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG

        print(f"Shape de X et Y : {imgs.shape} et {labels.shape}")
        print(f"Premier score d'image {imgs[0, :, :, :].shape} : ", labels[0])

        # testing_slice = [0.2, 0.4]  # [0, 0.2] => je prends les 20 % premiers
        # nb_dst_from_one_src = 64 * (5*5 + 3)  # 64 patches, 5 dist*5 lvl + 3 lvl contraste
        # OBSOLETE : ne marche pas si parfois il y a 3 niveaux de contrastes, parfois 4

        dataframe = pd.read_csv(path_csv)
        print("df", dataframe['image'])
        print("df uniques", pd.unique(dataframe['image']))
        uniques_names = pd.unique(dataframe['image'])
        unique_count = list(dataframe['image'].value_counts(sort=False))

        # Split testing_slice and train #

        # uniquesFIRST = np.arange(imgs.shape[0] / nb_dst_from_one_src, dtype=int)  # Obsolete, voire au dessus
        uniques = np.arange(len(uniques_names), dtype=int)

        uniques_slice_points = np.array(
            [round(len(uniques) * testing_slice[0]), round(len(uniques) * testing_slice[1])])

        # dst_slice_pointsFIRST = uniques_slice_points * nb_dst_from_one_src  # Obsolete, voire au dessus
        dst_slice_points = np.array([sum(unique_count[:uniques_slice_points[0]]) * NB_PATCH_PER_IMG, sum(
            unique_count[:uniques_slice_points[1]]) * NB_PATCH_PER_IMG])  # Tentative sans magic number

        imgs_shape = imgs.shape

        img_test = imgs[dst_slice_points[0]:dst_slice_points[1], ...]
        img_train = np.append(imgs[:dst_slice_points[0], ...], imgs[dst_slice_points[1]:, ...], axis=0)

        label_test = labels[dst_slice_points[0]:dst_slice_points[1], ...]
        label_train = np.append(labels[:dst_slice_points[0], ...], labels[dst_slice_points[1]:, ...], axis=0)

        if cst.PREPROCESS_ON_THE_FLY and cst.SEPARATE_NORM_TEST_NORM_TRAIN:

            indices_images = np.random.randint(img_test.shape[0], size=10)
            echantillon_images_before = np.empty((10, cst.input_shape[0], cst.input_shape[1], cst.input_shape[2]))
            for mini_i, value in enumerate(indices_images):
                echantillon_images_before[mini_i, :, :, :] = img_test[value, :, :, :]
            plot10Images(echantillon_images_before)

            print(f"Min/max of test dataset before norm : ({np.amin(img_test)}/{np.amax(img_test)})")
            print(f"Min/max of train dataset before norm : ({np.amin(img_train)}/{np.amax(img_train)})")

            print(f"Preprocessing ({cst.PREPROCESS_IMG}) on the fly...")
            img_train = Normalization.preprocess_images(img_train, cst.PREPROCESS_IMG)
            print("Done on train. Doing on test...")
            img_test = Normalization.preprocess_images(img_test, cst.PREPROCESS_IMG)
            print("Done.")

            echantillon_images_after = np.empty((10, cst.input_shape[0], cst.input_shape[1], cst.input_shape[2]))
            for mini_i, value in enumerate(indices_images):
                echantillon_images_after[mini_i, :, :, :] = img_test[value, :, :, :]
            plot10Images(echantillon_images_after)

            print(f"Min/max of test dataset after norm : ({np.amin(img_test)}/{np.amax(img_test)})")
            print(f"Min/max of train dataset after norm : ({np.amin(img_train)}/{np.amax(img_train)})")

        print(f"uniques : {uniques}")
        print(f"uniques_slice_points : {uniques_slice_points}")
        print(f"dst_slice_points : {dst_slice_points}")
        print(f"img_test {img_test.shape} + img_train {img_train.shape} = imgs {imgs_shape}")
        print(f"label_test {label_test.shape} + label_train {label_train.shape} = labels {labels.shape}")

        # Train #

        initializer = tf.keras.initializers.he_normal()

        # Creating the CNN model
        VGG16 = tf.keras.applications.VGG16(input_shape=cst.input_shape, weights=None, include_top=False)
        output = Model.regression_block(VGG16.output, initializer)
        custom_VGG16 = tf.keras.models.Model(inputs=VGG16.input, outputs=output)

        # print("MON VGG16")
        # custom_VGG16.summary()

        # Compile and configures the model for training.
        custom_VGG16.compile(optimizer=cst.optimizer, loss='mse',
                             metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

        # Configure checkpoints to save training

        if cst.SAVE_TRAINING:
            callbacks_arr = create_callbacks_fun(dir_logs, dir_weights, fold_no=0, save_name=name_log_weight)
            fit_callbacks = callbacks_arr
        else:
            fit_callbacks = None

        # Trains the model for a fixed number of epochs (iterations on the used dataset)

        print("Checkpoint 1")

        training_history = custom_VGG16.fit(x=img_train, y=label_train, batch_size=cst.batch_size,
                                            epochs=cst.epochs, validation_split=cst.VALID_BASE_PERCENTAGE, shuffle=True,
                                            callbacks=fit_callbacks)
        print("Checkpoint 2")

        # plot loss and val_loss along the training

        plot_training_history(training_history)

        print('[INFO] Evaluating/Predicting...')

        evaluation_score = custom_VGG16.evaluate(img_test, label_test, batch_size=16)  # 8 a l'origine

        predictions = custom_VGG16.predict(img_test, batch_size=16)

        print(f'loss = {evaluation_score[0]},  accuracy = {evaluation_score[1]}')
        print(f" Prédictions de scores : {predictions.flatten()} shape : {predictions.shape}")
        print(f" Vrais scores : {label_test.flatten()} shape : {label_test.shape}")
        plcc = pearsonr(predictions.flatten(), label_test.flatten())
        srocc = spearmanr(predictions.flatten(), label_test.flatten())
        print(f'PLCC = {plcc[0]}, SROCC = {srocc[0]}')

        # Scatter plot to vosualize the relation between the actual MOS and the
        # predicted one

        plot_test_prediction(label_test, predictions, name_predictions_arr)

        print(f"Saving predictions test to : {dir_predictions_arr}/{name_predictions_arr}")
        save_perf(predictions, label_test, dir_predictions_arr, name_predictions_arr)
