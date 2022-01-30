import numpy as np
import math
import tensorflow as tf
from model import Model
from utiles import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


def run(env):
    """This codes comes after Normalization and patching. The data is already preprocessed and saved in .npy files"""

    # Parameters #

    if env == "COLAB":
        import constants_for_colab as cst
    elif env == "LOCAL":
        import constants as cst
    else:
        raise ImportError

    NB_PATCH_PER_IMG = cst.DATA_AUGMENT_FACTOR ** 2

    # for iter in np.arange(0, 1, cst.TEST_BASE_PERCENTAGE)
    # testing_slice = [0, 0.2]

    # for iter in cst.ITERATION:
    # saving paths
    if cst.ITERATION == 'all':
        all_testing_slice = np.append(np.arange(0, 1, cst.TEST_BASE_PERCENTAGE), 1)
    else:
        all_testing_slice = np.append(np.arange(0, 1, cst.TEST_BASE_PERCENTAGE), 1)[cst.ITERATION: cst.ITERATION + 2]
    print("all_testing_slice : ", all_testing_slice)
    print("for : ", range(len(all_testing_slice) - 1))
    print("first testing slice : ", all_testing_slice[0:0+2])

    for i_iteration in range(len(all_testing_slice) - 1):

        testing_slice = all_testing_slice[i_iteration:i_iteration + 2]
        print(f"COMMENCEMENT ITERATION {i_iteration} ({testing_slice})")

        name_log_weight = f"iteration{i_iteration}"
        dir_predictions_arr = cst.ROOT_PROJECT / 'test_results' / cst.DATABASE_NAME / cst.PREPROCESS_IMG
        name_predictions_arr = f"{cst.epochs}epochs_iteration{i_iteration}"
        dir_logs = cst.LOGS_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG
        dir_weights = cst.SAVED_TRAINING_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG

        if cst.USE_GRAY:
            imgs = np.load(
                cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_GRAY.npy")
            labels = np.load(
                cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_GRAY_labels.npy")
        else:
            imgs = np.load(
                cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}.npy")
            labels = np.load(
                cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"CSIQ_{cst.PREPROCESS_IMG}_patch{64}_labels.npy")

        print(f"Shape de X et Y : {imgs.shape} et {labels.shape}")
        print(f"Premier score d'image {imgs[0, :, :, :].shape} : ", labels[0, 0, 0, 0])
        labels = np.squeeze(labels)
        print(f"Shape de X et Y after squeeze : {imgs.shape} et {labels.shape}")

        # testing_slice = [0.2, 0.4]  # [0, 0.2] => je prends les 20 % premiers
        # nb_dst_from_one_src = 64 * (5*5 + 3)  # 64 patches, 5 dist*5 lvl + 3 lvl contraste
        # OBSOLETE : ne marche si parfois il y a 3 niveaux de contrastes, parfois 4

        dataframe = pd.read_csv(cst.CSV_DATAFILE_DIR / cst.DATAFILE_NAME)
        print("df", dataframe['image'])
        print("df uniques", np.unique(dataframe['image']))
        uniques_names, unique_count = np.unique(dataframe['image'], return_counts=True)

        # Split testing_slice and train #

        # uniquesFIRST = np.arange(imgs.shape[0] / nb_dst_from_one_src, dtype=int)  # Obsolete, voire au dessus
        uniques = np.arange(len(uniques_names), dtype=int)

        uniques_slice_points = np.array(
            [round(len(uniques) * testing_slice[0]), round(len(uniques) * testing_slice[1])])

        # dst_slice_pointsFIRST = uniques_slice_points * nb_dst_from_one_src  # Obsolete, voire au dessus
        dst_slice_points = np.array([sum(unique_count[:uniques_slice_points[0]]) * NB_PATCH_PER_IMG, sum(
            unique_count[:uniques_slice_points[1]]) * NB_PATCH_PER_IMG])  # Tentative sans magic number

        img_test = imgs[dst_slice_points[0]:dst_slice_points[1], ...]
        img_train = np.append(imgs[:dst_slice_points[0], ...], imgs[dst_slice_points[1]:, ...], axis=0)

        label_test = labels[dst_slice_points[0]:dst_slice_points[1], ...]
        label_train = np.append(labels[:dst_slice_points[0], ...], labels[dst_slice_points[1]:, ...], axis=0)

        print(f"uniques : {uniques}")
        print(f"uniques_slice_points : {uniques_slice_points}")
        print(f"dst_slice_points : {dst_slice_points}")
        print(f"img_test {img_test.shape} + img_train {img_train.shape} = imgs {imgs.shape}")
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

        if cst.USE_GRAY:
            plt.imshow(img_train[0, :, :, 0], cmap='gray');
            plt.show()
            plt.imshow(img_train[1, :, :, 0], cmap='gray');
            plt.show()
            plt.imshow(img_train[2, :, :, 0], cmap='gray');
            plt.show()
        else:
            plt.imshow(img_train[0, :, :, :]);
            plt.show()
            plt.imshow(img_train[1, :, :, :]);
            plt.show()
            plt.imshow(img_train[2, :, :, :]);
            plt.show()

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
