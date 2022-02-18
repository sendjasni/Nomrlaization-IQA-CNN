import tensorflow as tf


class Model:

    @staticmethod
    def regression_block(features, initializer):
        # x = tf.keras.layers.Flatten()(features)
        x = tf.keras.layers.GlobalAveragePooling2D()(features)

        # x = tf.keras.layers.Dense(1024, activation="relu",
        #                           kernel_initializer=initializer)(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(
            1, activation='linear', kernel_initializer=initializer)(x)

        return x

    @staticmethod
    def regression_block_first(features, initializer):
        x = tf.keras.layers.Flatten()(features)
        # x = tf.keras.layers.GlobalAveragePooling2D()(features)

        x = tf.keras.layers.Dense(1024, activation="relu",
                                  kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(
            1, activation='linear', kernel_initializer=initializer)(x)

        return x

    @staticmethod
    def regression_block_new(features, initializer):
        x = tf.keras.layers.Flatten()(features)
        x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(1, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Activation(tf.keras.activations.linear)(x)
        return x

    @staticmethod
    def create_model_v1(input_shape, initializer, filters=(64, 128, 256, 512)):
        """premier modele utilisé"""
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

    @staticmethod
    def create_model_v2(input_shape, pretrained=False):
        """2e version du modele, VGG16 + regression_block()"""
        initializer = tf.keras.initializers.he_normal()
        if pretrained:
            VGG16 = tf.keras.applications.VGG16(input_shape=input_shape, weights="imagenet", include_top=False)
        else:
            VGG16 = tf.keras.applications.VGG16(input_shape=input_shape, weights=None, include_top=False)
        output = Model.regression_block(VGG16.output, initializer)

        return tf.keras.models.Model(inputs=VGG16.input, outputs=output)

    @staticmethod
    def create_model_v3(input_shape, verbose=False):
        """3e version du modele"""
        # input
        initializer = tf.keras.initializers.he_normal()

        input_ = tf.keras.layers.Input(shape=input_shape)
        # 1st Conv Block

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(input_)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        # 2nd Conv Block

        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3,
                                   padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3,
                                   padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        # 3rd Conv block

        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                                   padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                                   padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=3,
                                   padding='same', kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)
        # 4th Conv block

        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.01), kernel_initializer=initializer)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='same')(x)

        # # 5th Conv block

        # x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
        # x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
        # x = Conv2D (filters =512, kernel_size =3, padding ='same', activation='relu')(x)
        # x = MaxPool2D(pool_size =2, strides =2, padding ='same')(x)
        # # Fully connected layers

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dense(1024, activation="relu",
        #                             kernel_initializer=initializer)(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        x = tf.keras.layers.Dense(
            1, activation='linear', kernel_initializer=initializer)(x)
        # creating the model

        iqa_vgg16 = tf.keras.Model(inputs=input_, outputs=x)
        iqa_vgg16.summary()
        return iqa_vgg16
