from pathlib import Path
import tensorflow as tf

# Folder names and File names : (of Folders, Normalizations, etc.)
DATABASE_NAME = "CLIVE"  # TID2013 / CSIQ / KADID10K
# Preprocessing
PREPROCESS_IMG = 'nothing'  # scale_0_1, centering, centering_reduced, standardization, LCN, LRN, histogram_eq, CLAHE, ZCAwhiten, simplified_whitening
PREPROCESS_ON_THE_FLY = True  # Preprocess from
SEPARATE_NORM_TEST_NORM_TRAIN = True  # si l'algorithme de preprocessing se fait sur tout un set et pas indépendament sur chaque img, il vaut mieux mettre ça à True
# Peut prendre + de temps, puisqu'à chaque Segmentation Test/Train de la base, les 2 sets devront être normalisés

# Paths :

ROOT_PROJECT = Path("./")
ROOT_DATABASE = ROOT_PROJECT / "database"
# ROOT_DST_IMG = ROOT_DATABASE / DATABASE_NAME / "dst_imgs"
# CSV_DATAFILE_DIR = ROOT_DATABASE / "csv_datafile"
SAVED_TRAINING_DIR = ROOT_PROJECT / "saved_training"
LOGS_DIR = ROOT_PROJECT / "training_logs"

# User options :
FIX_RANDOM = True
SAVE_TRAINING = True
ENABLE_PATCH_DATA_AUGMENTATION = True
USE_GRAY = True
WEIGHTS_IMAGENET = True
TEST_BASE_PERCENTAGE = 0.2  # Pourcentage de test/train (la base de validation sera prélevée dans la base de train)
VALID_BASE_PERCENTAGE = 0.2  # Pourcentage de valid/train (la base de validation sera prélevée dans la base de train)
ITERATION = 'all'  # 0, 1, 2, 3, 4, 'all'

# Hyper-parameters for the training process
if DATABASE_NAME == 'CSIQ':
    input_shape_before_patch = (512, 512, 3) if not USE_GRAY else (512, 512, 1)  # input shape of the model depending on the shape of input images, so be carefull.
elif DATABASE_NAME == 'TID2013':
    input_shape_before_patch = (384, 512, 3) if not USE_GRAY else (384, 512, 1)  # input shape of the model depending on the shape of input images, so be carefull.
elif DATABASE_NAME == 'LIVE2':
    input_shape_before_patch = (384, 448, 3) if not USE_GRAY else (384, 448, 1)  # input shape of the model depending on the shape of input images, so be carefull.
elif DATABASE_NAME == 'CLIVE':
    input_shape_before_patch = (448, 448, 3) if not USE_GRAY else (448, 448, 1)  # input shape of the model depending on the shape of input images, so be carefull.

if WEIGHTS_IMAGENET:
    input_shape_before_patch = (input_shape_before_patch[0], input_shape_before_patch[1], 3)

if ENABLE_PATCH_DATA_AUGMENTATION:
    input_shape = (64, 64, input_shape_before_patch[2])
    DATA_AUGMENT_FACTOR = (round(input_shape_before_patch[0]/input_shape[0]), round(input_shape_before_patch[1]/input_shape[1]))
else:
    input_shape = input_shape_before_patch
    DATA_AUGMENT_FACTOR = (1, 1)

# input_shape = (input_shape_before_patch[0]//DATA_AUGMENT_FACTOR, input_shape_before_patch[1]//DATA_AUGMENT_FACTOR, input_shape_before_patch[2])  # input shape of the model depending on the shape of input images, so be carefull.
# You need to get this from one of the images. It is preferable to start with a shape of (256, 256, C),
# C is the number of channels (If RGB then C=3, else C=1)
batch_size = 32  # mini-batch size
epochs = 300  # number of epochs.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=1e-4,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8,
    amsgrad=True,  #modelv2 = False
    name="Adam",
)
patience = 10
# tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-4 / epochs)  # optimizer for the model's training
# a l'origine learning rate = 1e-3