from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam
from metrics import dice_coef, dice_coef_loss
from keras.losses import binary_crossentropy


class ThreeDUnet():

    def __init__(self, model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        self.model = model


    def create_model(self, img_shape):

        concat_axis = 4

        inputs = layers.Input(shape=img_shape)

        conv1 = layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')(inputs)
        conv2 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(conv1)
        maxpool1 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv2)
        #bn_1 = layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',)
        #                   gamma_initializer='ones', moving_mean_initializer='zeros',
        #                   moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None,
        #                   beta_constraint=None, gamma_constraint=None)

        conv3 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(maxpool1)
        conv4 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(conv3)
        maxpool2 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv4)

        conv5 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(maxpool2)
        conv6 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv5)
        maxpool3 = layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv6)

        conv7 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(maxpool3)
        conv8 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv7)

        up_samp1 = layers.UpSampling3D(size=(2, 2, 2))(conv8)
        conv9 = layers.Conv3D(512, kernel_size=2, padding='same', activation='relu')(up_samp1)
        concat_1 = layers.concatenate([conv6, conv9], axis=concat_axis)

        conv10 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(concat_1)
        conv11 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv10)

        up_samp2 = layers.UpSampling3D(size=(2, 2, 2))(conv11)
        conv12 = layers.Conv3D(256, kernel_size=2, padding='same', activation='relu')(up_samp2)
        concat_2 = layers.concatenate([conv4, conv12], axis=concat_axis)

        conv13 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(concat_2)
        conv14 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(conv13)

        up_samp3 = layers.UpSampling3D(size=(2, 2, 2))(conv14)
        conv15 = layers.Conv3D(128, kernel_size=2, padding='same', activation='relu')(up_samp3)
        concat_3 = layers.concatenate([conv2, conv15], axis=concat_axis)

        conv16 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(concat_3)
        conv17 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(conv16)

        conv18 = layers.Conv3D(1, kernel_size=1, padding='same', activation='softmax')(conv17)

        model = models.Model(inputs=inputs, outputs=conv18)

        model.compile(optimizer=Adam(lr=0.01), loss=binary_crossentropy,
                      metrics=[dice_coef, binary_crossentropy])

        model.summary()

        return model


    def save_specs(self, specs_path, fit_specs):
        """Guardar todas las caracterIsticas del modelo y las opciones del
        metodo fit() en un fichero txt y las caracterIsticas del modelo de
        la librerIa keras en formato json en la carpeta de logs.
        """

        with open(specs_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()

        fit_specs_file = specs_path[:-4] + 'fit_specs.txt'

        with open(fit_specs_file, 'w') as fit_file:
            for key, value in fit_specs.items():
                fit_file.write(key + ': ' + str(value) + '\n')

    def create_folders(self, training_name, base_path):
        """
        Metodo para crear las siguientes carpetas y directorios:
                model_path: carpeta para guardar los pesos del modelo
                weights_path: directorio de los pesos del modelo concreto con
                              el formato de fichero model_x.hdf5, donde x es la
                              version del modelo.
                              Si una version concreta existe, se crea un
                              fichero nuevo con el siguiente numero de version
                log_path: carpeta donde guardar los logs de TensorBoard y las
                          caracterIsticas del modelo
        """

        model_path = base_path + "/models/" + training_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        v = 0
        weights_path = model_path + "/model_0.hdf5"
        if os.path.exists(weights_path):
            try:
                v = int(weights_path.split("_")[-1].replace(".hdf5", "")) + 1
            except ValueError:
                v = 1
            weights_path = model_path + "/model_{}.hdf5".format(v)

        log_path = base_path + "/logs/" + training_name + '/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        specs_path = log_path + "/specs_{}.txt".format(v)

        return {"log_path": log_path, "weights_path": weights_path,
                "specs_path": specs_path}

    def train(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        paths = self.create_folders(training_name, base_path)

        checkpointer = ModelCheckpoint(filepath=paths["weights_path"],
                                       save_best_only=False,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=paths["log_path"],
                                           histogram_freq=0,
                                           batch_size=batch_size,
                                           write_graph=False,
                                           write_grads=True,
                                           write_images=False,
                                           embeddings_freq=0,
                                           embeddings_layer_names=None,
                                           embeddings_metadata=None)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size)

        fit_specs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'test_size': test_size

        }
        self.save_specs(paths['specs_path'], fit_specs)

        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=1)






