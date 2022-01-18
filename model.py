import tensorflow as tf


class Model:

    @staticmethod
    def regression_block(features, initializer):
        x = tf.keras.layers.Dense(1024, activation="relu",
                                  kernel_initializer=initializer)(features)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, activation="relu",
                                  kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Dense(
            1, activation='linear', kernel_initializer=initializer)(x)

        return x

    @staticmethod
    def create_model(input_shape, initializer, filters=(64, 128, 256, 512)):
        chan_dim = -1
        inputs = tf.keras.layers.Input(shape=input_shape)

        for (i, f) in enumerate(filters):
            if i == 0:
                x = inputs
            x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
                3, 3), padding="same", kernel_initializer=initializer)(x)
            x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

            x = tf.keras.layers.Conv2D(filters=f, kernel_size=(
                3, 3), padding="same", kernel_initializer=initializer)(x)
            x = tf.keras.layers.BatchNormalization(axis=chan_dim)(x)
            x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)

            max_p = tf.keras.layers.MaxPooling2D(
                pool_size=(2, 2), strides=(2, 2))(x)
            avrg_p = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2))(x)
            # min_p = tf.keras.layers.MaxPooling2D( # pas sur de si je devais ajouter ca
            #     pool_size=(2, 2), strides=(2, 2))(x)

            x = tf.keras.layers.concatenate([avrg_p, max_p])
            # x = tf.keras.layers.concatenate([avrg_p, max_p, min_p])

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        output = Model.regression_block(x, initializer)

        iqa_model = tf.keras.models.Model(inputs=inputs, outputs=output)
        return iqa_model
