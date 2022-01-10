import tensorflow as tf
from timeit import default_timer as timer

class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)

def create_callbacks_fun(path, fold_no, Normalization):

    log_file = path + '/Fold_' + str(fold_no) + '.log'

    csv_loger = tf.keras.callbacks.CSVLogger(
        log_file, separator=",", append=False)

    filepath = path + '/weights_' + Normalization + '.hdf5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", mode="min",
                                                    save_best_only=True, save_weights_only=True, save_freq='epoch', verbose=1)
    train_tile_cb = TimingCallback()
    callbacks = [
        csv_loger,
        train_tile_cb,
    ]

    return callbacks