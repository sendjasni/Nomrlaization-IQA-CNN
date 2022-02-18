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

    print("Database name : ", cst.DATABASE_NAME)
    if cst.DATABASE_NAME == "TID2013":
        path_csv = cst.ROOT_DATABASE / cst.DATABASE_NAME / "mos.csv"
    else:
        raise Exception('database "'+cst.DATABASE_NAME+'" not found.')

    NB_PATCH_PER_IMG = cst.DATA_AUGMENT_FACTOR[0] * cst.DATA_AUGMENT_FACTOR[1]

    gray_str = '_GRAY' if cst.USE_GRAY else ''
    preprocess_npy_str = 'nothing' if cst.PREPROCESS_ON_THE_FLY else cst.PREPROCESS_IMG
    path_dataset = cst.ROOT_DATABASE / cst.DATABASE_NAME / "normalized_dst_imgs" / f"{cst.DATABASE_NAME}_{preprocess_npy_str}_patch{64}{gray_str}"

    print(f"Loading database... ({path_dataset}.npy)")
    img_train = np.load(f"{path_dataset}.npy")
    label_train = np.load(f"{path_dataset}_labels.npy")
    label_train = np.squeeze(label_train)
    print("Done.")

    if cst.USE_GRAY:
        first_img_before_norm = img_train[0, :, :, 0]
        plt.imshow(first_img_before_norm, cmap='gray'); plt.show()
    else:
        first_img_before_norm = img_train[0, :, :, :]
        plt.imshow(first_img_before_norm); plt.show()
    print(f"Min/max of dataset before norm : ({np.amin(img_train)}/{np.amax(img_train)})")

    print(f"Preprocessing ({cst.PREPROCESS_IMG}) on the fly...")
    img_train = Normalization.preprocess_images(img_train, cst.PREPROCESS_IMG)
    print("Done.")

    if cst.USE_GRAY:
        first_img_after_norm = img_train[0, :, :, 0]
        plt.imshow(first_img_after_norm, cmap='gray'); plt.show()
    else:
        first_img_after_norm = img_train[0, :, :, :]
        plt.imshow(first_img_after_norm); plt.show()
    print(f"Min/max of dataset after norm : ({np.amin(img_train)}/{np.amax(img_train)})")

    i_iteration = "_entireset"

    # i_iteration = i_slice + (round(testing_slice[0] / cst.TEST_BASE_PERCENTAGE))
    print(f"COMMENCEMENT ITERATION {i_iteration}")

    name_log_weight = f"{cst.PREPROCESS_IMG}_iteration{i_iteration}"
    dir_predictions_arr = cst.ROOT_PROJECT / 'test_results' / cst.DATABASE_NAME / cst.PREPROCESS_IMG
    name_predictions_arr = f"{cst.epochs}epochs_iteration{i_iteration}{gray_str}"
    dir_logs = cst.LOGS_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG
    dir_weights = cst.SAVED_TRAINING_DIR / cst.DATABASE_NAME / cst.PREPROCESS_IMG

    print(f"Shape de X et Y : {img_train.shape} et {label_train.shape}")
    print(f"Premier score d'image {img_train[0, :, :, :].shape} : ", label_train[0])

    # Train #

    # model = Model.create_model_v2(cst.input_shape)
    # model = Model.create_model_v2(cst.input_shape, pretrained=cst.WEIGHTS_IMAGENET)
    model = Model.create_model_v3(cst.input_shape)

    # print("MON VGG16")
    # custom_VGG16.summary()

    # Compile and configures the model for training.
    model.compile(optimizer=cst.optimizer, loss='mse',
                         metrics=tf.keras.metrics.RootMeanSquaredError(name='rmse'))

    # Configure checkpoints to save training

    if cst.SAVE_TRAINING:
        callbacks_arr = create_callbacks_fun(dir_logs, dir_weights, fold_no=0, save_name=name_log_weight, patience=cst.patience, save_weights=True)
        fit_callbacks = callbacks_arr
    else:
        fit_callbacks = None

    # Trains the model for a fixed number of epochs (iterations on the used dataset)

    print("Checkpoint 1")

    training_history = model.fit(x=img_train, y=label_train, batch_size=cst.batch_size,
                                        epochs=cst.epochs, validation_split=cst.VALID_BASE_PERCENTAGE, shuffle=True,
                                        callbacks=fit_callbacks)
    print("Checkpoint 2")

    # plot loss and val_loss along the training

    plot_training_history(training_history)

    print('[INFO] Evaluating/Predicting...')
