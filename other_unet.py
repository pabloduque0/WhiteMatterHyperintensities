import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from contextlib import redirect_stdout
from keras.models import load_model


class TwoDUnet():

    def __init__(self, model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        self.model = model

    def create_model(self, img_shape=None):
        inputs = Input(img_shape)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

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
                                           write_graph=True,
                                           write_grads=True,
                                           write_images=True,
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
                       verbose=2)