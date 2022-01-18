from pathlib import Path
import tensorflow as tf

# Paths :

ROOT_DATABASE = Path("./database")
CSV_DATAFILE_DIR = ROOT_DATABASE / "csv_datafile"
SAVED_TRAINING_DIR = Path("./saved_training")
LOGS_DIR = Path("./training_logs")
DATAFILE_NAME = 'CSIQ_DMOS_allAWGN.csv'

# User options :

POSSIBLE_PREPROCESSES_IMG = ['scale_0_1', 'centering', 'standardization', 'LCN', 'LRN', 'histogram eq', 'CLA hist eq', 'ZCA whiten']
PREPROCESS_IMG = 'CLA hist eq'
# NO_TRAINING = False  # just recover older training and test it
FIX_RANDOM = True
SAVE_TRAINING = True
ENABLE_PATCH_DATA_AUGMENTATION = True
DATA_AUGMENT_FACTOR = 2  # if 2 => one img divided by 2 in height and by 2 in width (4 patchs)
TEST_BASE_PERCENTAGE = 0.2  # Pourcentage de test/train (la base de validation sera prélevée dans la base de train)
VALID_BASE_PERCENTAGE = 0.2  # Pourcentage de valid/train (la base de validation sera prélevée dans la base de train)

# Hyper-parameters for the training process

input_shape_before_data_augment = (512, 512, 3)  # input shape of the model depending on the shape of input images, so be carefull.
input_shape = (512//DATA_AUGMENT_FACTOR, 512//DATA_AUGMENT_FACTOR, 3)  # input shape of the model depending on the shape of input images, so be carefull.
# You need to get this from one of the images. It is preferable to start with a shape of (256, 256, C),
# C is the number of channels (If RGB then C=3, else C=1)
batch_size = 16  # mini-batch size
epochs = 100  # number of epochs.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / epochs)  # optimizer for the model's training
