from pathlib import Path
import tensorflow as tf

# Paths :

ROOT_DATABASE = Path("./database")
CSV_DATAFILE_DIR = ROOT_DATABASE / "csv_datafile"
SAVED_TRAINING_DIR = Path("./saved_training")
LOGS_DIR = Path("./training_logs")
DATAFILE_NAME = 'CSIQ_DMOS_allAWGN.csv'

# User options :

POSSIBLE_PREPROCESSES_IMG = ['SCALE_0_1', 'CENTERING', 'STANDARD_SIGMA']
PREPROCESS_IMG = ''
# NO_TRAINING = False  # just recover older training and test it
FIX_RANDOM = True
SAVE_TRAINING = True
ENABLE_PATCH_DATA_AUGMENTATION = True
TEST_BASE_PERCENTAGE = 0.2  # Pourcentage de test/train (la base de validation sera prélevée dans la base de train)
VALID_BASE_PERCENTAGE = 0.2  # Pourcentage de valid/train (la base de validation sera prélevée dans la base de train)

# Hyper-parameters for the training process

input_shape = (512, 512, 3)  # input shape of the model depending on the shape of input images, so be carefull.
# You need to get this from one of the images. It is preferable to start with a shape of (256, 256, C),
# C is the number of channels (If RGB then C=3, else C=1)
batch_size = 16  # mini-batch size
epochs = 100  # number of epochs.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / epochs)  # optimizer for the model's training
