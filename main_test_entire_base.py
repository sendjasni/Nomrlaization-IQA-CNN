import numpy as np
import math
import tensorflow as tf
from model import Model
from utiles import *
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from normalization import Normalization
from numbers import Number


def run(env, cst):
    """This codes comes after patching. The data is already patched and saved in .npy files
    if this is run from local computer, don't precise cst argument
    """

    print("Database name : ", cst.DATABASE_NAME)
    if cst.DATABASE_NAME == "CLIVE":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "mos.csv"
    else:
        raise Exception('database "'+cst.DATABASE_NAME+'" not found.')

    # for iter in np.arange(0, 1, cst.TEST_BASE_PERCENTAGE)
    # testing_slice = [0, 0.2]

    # for iter in cst.ITERATION:
    # saving paths

    gray_str = '_GRAY' if cst.USE_GRAY else ''
    preprocess_npy_str = 'nothing' if cst.PREPROCESS_ON_THE_FLY else cst.PREPROCESS_IMG
    path_dataset = cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{preprocess_npy_str}_patch{64}{gray_str}"

    print(f"Loading database... ({path_dataset}.npy)")
    img_test = np.load(f"{path_dataset}.npy")
    label_test = np.load(f"{path_dataset}_labels.npy")
    label_test = np.squeeze(label_test)
    print("Done.")
    if cst.USE_GRAY:
        first_img_before_norm = img_test[0, :, :, 0]
        plt.imshow(first_img_before_norm, cmap='gray'); plt.show()
    else:
        first_img_before_norm = img_test[0, :, :, :]
        plt.imshow(first_img_before_norm); plt.show()
    print(f"Min/max of dataset before norm : ({np.amin(img_test)}/{np.amax(img_test)})")

    print(f"Preprocessing ({cst.PREPROCESS_IMG}) on the fly...")
    img_test = Normalization.preprocess_images(img_test, cst.PREPROCESS_IMG)
    print("Done.")

    if cst.USE_GRAY:
        first_img_after_norm = img_test[0, :, :, 0]
        plt.imshow(first_img_after_norm, cmap='gray'); plt.show()
    else:
        first_img_after_norm = img_test[0, :, :, :]
        plt.imshow(first_img_after_norm); plt.show()
    print(f"Min/max of dataset after norm : ({np.amin(img_test)}/{np.amax(img_test)})")

    # Create model and test #

    # model = Model.create_model_v2(cst.input_shape)
    # model = Model.create_model_v2(cst.input_shape, pretrained=cst.WEIGHTS_IMAGENET)
    model = Model.create_model_v3(cst.input_shape)

    # Load weights
    i_iteration = "_entireset"
    dir_weights = cst.SAVED_TRAINING_DIR / "TID2013" / cst.PREPROCESS_IMG
    name_log_weight = f"{cst.PREPROCESS_IMG}_iteration{i_iteration}"
    dir_predictions_arr = cst.ROOT_PROJECT / 'test_results' / cst.DATABASE_NAME / cst.PREPROCESS_IMG
    name_predictions_arr = f"{cst.epochs}epochs_iteration{i_iteration}{gray_str}"
    model.load_weights(dir_weights / (name_log_weight+".hdf5"))

    # print("MON VGG16")
    # custom_VGG16.summary()

    # Compile and configures the model for training.
    model.compile(optimizer=cst.optimizer, loss='mse',
                         metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

    print('[INFO] Evaluating/Predicting...')

    evaluation_score = model.evaluate(img_test, label_test, batch_size=16)  # 8 a l'origine

    predictions = model.predict(img_test, batch_size=16)

    print(f'loss = {evaluation_score[0]},  accuracy = {evaluation_score[1]}')
    print(f" Prédictions de scores : {predictions.flatten()} shape : {predictions.shape}")
    print(f" Vrais scores : {label_test.flatten()} shape : {label_test.shape}")
    plcc = pearsonr(predictions.flatten(), label_test.flatten())
    srocc = spearmanr(predictions.flatten(), label_test.flatten())
    print(f'PLCC = {plcc[0]}, SROCC = {srocc[0]}')

    # Scatter plot to vosualize the relation between the actual MOS and the
    # predicted one

    fig = plot_test_prediction(label_test, predictions, name_predictions_arr)
    print(f"Saving plot test to : {dir_predictions_arr}/{name_predictions_arr}")
    fig.savefig(dir_predictions_arr/name_predictions_arr)

    print(f"Saving predictions test to : {dir_predictions_arr}/{name_predictions_arr}")
    save_perf(predictions, label_test, dir_predictions_arr, name_predictions_arr)

