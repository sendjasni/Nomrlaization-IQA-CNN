from pathlib import Path
import tensorflow as tf

# Names : (of Folders, Normalizations, etc.)

DATABASE_NAME = "CSIQ"  # name of the folder
PREPROCESS_IMG = 'histogram_eq'  # scale_0_1, centering, centering_reduced, standardization, LCN, LRN, histogram_eq, CLAHE, ZCAwhiten, simplified_whitening
DATAFILE_NAME = 'CSIQ_DMOS.csv'

# Paths :

ROOT_PROJECT = Path("./")
ROOT_DATABASE = ROOT_PROJECT / "database"
ROOT_DST_IMG = ROOT_DATABASE / DATABASE_NAME / "dst_imgs"
CSV_DATAFILE_DIR = ROOT_DATABASE / "csv_datafile"
SAVED_TRAINING_DIR = ROOT_PROJECT / "saved_training"
LOGS_DIR = ROOT_PROJECT / "training_logs"

# User options :

# NO_TRAINING = False  # just recover older training and test it
FIX_RANDOM = True
SAVE_TRAINING = True
ENABLE_PATCH_DATA_AUGMENTATION = True
USE_GRAY = True
DATA_AUGMENT_FACTOR = 8  # if 2 => one img divided by 2 in height and by 2 in width (4 patchs)
TEST_BASE_PERCENTAGE = 0.2  # Pourcentage de test/train (la base de validation sera prélevée dans la base de train)
VALID_BASE_PERCENTAGE = 0.2  # Pourcentage de valid/train (la base de validation sera prélevée dans la base de train)
# ITERATION = 0  # 0, 1, 2, 3, 4

# Hyper-parameters for the training process

input_shape_before_patch = (512, 512, 3) if not USE_GRAY else (512, 512, 1) # input shape of the model depending on the shape of input images, so be carefull.
input_shape = (input_shape_before_patch[0]//DATA_AUGMENT_FACTOR, input_shape_before_patch[1]//DATA_AUGMENT_FACTOR, input_shape_before_patch[2])  # input shape of the model depending on the shape of input images, so be carefull.
# You need to get this from one of the images. It is preferable to start with a shape of (256, 256, C),
# C is the number of channels (If RGB then C=3, else C=1)
batch_size = 16  # mini-batch size
epochs = 300  # number of epochs.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    amsgrad=False,
    name="Adam",
)
# tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / epochs)  # optimizer for the model's training
# a l'origine learning rate = 1e-3