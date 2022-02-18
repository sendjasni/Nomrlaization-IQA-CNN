import numpy as np
import math
import tensorflow as tf
from model import Model
from utiles import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from normalization import Normalization
from numbers import Number


def run(env, cst=None):
    """This codes comes after patching. The data is already patched and saved in .npy files
    if this is run from local computer, don't precise cst argument
    """

    # Parameters #
    if env == "LOCAL":
        import constants as cst
    elif not cst:
        raise ImportError

    if cst.FIX_RANDOM:
        apply_seed(seed=111)  # Controls and fix the randomness of numpy/python/tensorflow (shuffling, etc.)

    print("Database name : ", cst.DATABASE_NAME)
    if cst.DATABASE_NAME == "CSIQ":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "CSIQ_DMOS.csv"
    elif cst.DATABASE_NAME == "TID2013":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "mos_with_names.csv"
    elif cst.DATABASE_NAME == "LIVE2":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "dmos.csv"
    else:
        raise Exception('database "'+cst.DATABASE_NAME+'" not found.')

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
    preprocess_npy_str = 'nothing' if cst.PREPROCESS_ON_THE_FLY else cst.PREPROCESS_IMG
    path_dataset = cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{preprocess_npy_str}_patch{64}{gray_str}"

    if cst.PREPROCESS_ON_THE_FLY and not cst.SEPARATE_NORM_TEST_NORM_TRAIN:
        print(f"Loading database... ({path_dataset}.npy)")
        imgs = np.load(f"{path_dataset}.npy")
        labels = np.load(f"{path_dataset}_labels.npy")
        labels = np.squeeze(labels)
        print("Done.")
        if cst.USE_GRAY:
            first_img_before_norm = imgs[0, :, :, 0]
            plt.imshow(first_img_before_norm, cmap='gray'); plt.show()
        else:
            first_img_before_norm = imgs[0, :, :, :]
            plt.imshow(first_img_before_norm); plt.show()
        print(f"Min/max of dataset before norm : ({np.amin(imgs)}/{np.amax(imgs)})")

        print(f"Preprocessing ({cst.PREPROCESS_IMG}) on the fly...")
        imgs = Normalization.preprocess_images(imgs, cst.PREPROCESS_IMG)
        print("Done.")

        if cst.USE_GRAY:
            first_img_after_norm = imgs[0, :, :, 0]
            plt.imshow(first_img_after_norm, cmap='gray'); plt.show()
        else:
            first_img_after_norm = imgs[0, :, :, :]
            plt.imshow(first_img_after_norm); plt.show()
        print(f"Min/max of dataset after norm : ({np.amin(imgs)}/{np.amax(imgs)})")

    for i_iteration in range(len(all_testing_slice) - 1):

        if cst.PREPROCESS_ON_THE_FLY and cst.SEPARATE_NORM_TEST_NORM_TRAIN:
            print(f"Loading database... ({path_dataset}.npy)")
            imgs = np.load(f"{path_dataset}.npy")
            labels = np.load(f"{path_dataset}_labels.npy")
            labels = np.squeeze(labels)
            # pd.DataFrame(labels).to_csv('verif_dmos.csv', index=False, header=['dmos'])
            print("Done.")

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
        print("dst slice points : ", dst_slice_points)
        imgs_shape = imgs.shape

        img_test = imgs[dst_slice_points[0]:dst_slice_points[1], ...]
        img_train = np.append(imgs[:dst_slice_points[0], ...], imgs[dst_slice_points[1]:, ...], axis=0)
        if cst.SEPARATE_NORM_TEST_NORM_TRAIN:
            del imgs

        label_test = labels[dst_slice_points[0]:dst_slice_points[1], ...]
        label_train = np.append(labels[:dst_slice_points[0], ...], labels[dst_slice_points[1]:, ...], axis=0)
        if cst.PREPROCESS_ON_THE_FLY and cst.SEPARATE_NORM_TEST_NORM_TRAIN:
            if cst.USE_GRAY:
                first_img_before_norm = img_test[0, :, :, 0]
                plt.imshow(first_img_before_norm, cmap='gray'); plt.show()
            else:
                first_img_before_norm = img_test[0, :, :, :]
                plt.imshow(first_img_before_norm); plt.show()
            print(f"Min/max of test dataset before norm : ({np.amin(img_test)}/{np.amax(img_test)})")
            print(f"Min/max of train dataset before norm : ({np.amin(img_train)}/{np.amax(img_train)})")

            print(f"Preprocessing ({cst.PREPROCESS_IMG}) on the fly...")
            img_train = Normalization.preprocess_images(img_train, cst.PREPROCESS_IMG)
            print("Done on train. Doing on test...")
            img_test = Normalization.preprocess_images(img_test, cst.PREPROCESS_IMG)

            if cst.USE_GRAY:
                first_img_after_norm = img_test[0, :, :, 0]
                plt.imshow(first_img_after_norm, cmap='gray'); plt.show()
            else:
                first_img_after_norm = img_test[0, :, :, :]
                plt.imshow(first_img_after_norm); plt.show()
            print(f"Min/max of test dataset after norm : ({np.amin(img_test)}/{np.amax(img_test)})")
            print(f"Min/max of train dataset after norm : ({np.amin(img_train)}/{np.amax(img_train)})")

        print(f"uniques : {uniques}")
        print(f"uniques_slice_points : {uniques_slice_points}")
        print(f"dst_slice_points : {dst_slice_points}")
        print(f"img_test {img_test.shape} + img_train {img_train.shape} = imgs {imgs_shape}")
        print(f"label_test {label_test.shape} + label_train {label_train.shape} = labels {labels.shape}")

        # Train #

        # model = Model.create_model_v2(cst.input_shape)
        model = Model.create_model_v2(cst.input_shape, pretrained=cst.WEIGHTS_IMAGENET)
        # model = Model.create_model_v3(cst.input_shape)

        # print("MON VGG16")
        # custom_VGG16.summary()

        # Compile and configures the model for training.
        model.compile(optimizer=cst.optimizer, loss='mse',
                             metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

        # Configure checkpoints to save training

        callbacks_arr = create_callbacks_fun(dir_logs, dir_weights, fold_no=0, save_name=name_log_weight, patience=cst.patience, save_weights=cst.SAVE_TRAINING)
        fit_callbacks = callbacks_arr

        # Trains the model for a fixed number of epochs (iterations on the used dataset)

        print("Checkpoint 1")

        if cst.USE_GRAY:
            plt.imshow(img_test[0, :, :, 0], cmap='gray'); plt.show()
            plt.imshow(img_test[1, :, :, 0], cmap='gray'); plt.show()
            plt.imshow(img_test[2, :, :, 0], cmap='gray'); plt.show()
        else:
            plt.imshow(img_test[0, :, :, :]); plt.show()
            plt.imshow(img_test[1, :, :, :]); plt.show()
            plt.imshow(img_test[2, :, :, :]); plt.show()

        if cst.USE_GRAY:
            img_train = triplicate_gray_channel(img_train)

        training_history = model.fit(x=img_train, y=label_train, batch_size=cst.batch_size,
                                            epochs=cst.epochs, validation_split=cst.VALID_BASE_PERCENTAGE, shuffle=True,
                                            callbacks=fit_callbacks)
        del img_train
        print("Checkpoint 2")

        # plot loss and val_loss along the training

        plot_training_history(training_history)

        print('[INFO] Evaluating/Predicting...')

        if cst.USE_GRAY:
            img_test = triplicate_gray_channel(img_test)
        evaluation_score = model.evaluate(img_test, label_test, batch_size=16)  # 8 a l'origine

        predictions = model.predict(img_test, batch_size=16)
        del img_test

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

