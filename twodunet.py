from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss, weighted_crossentropy
from keras.losses import binary_crossentropy
import cv2

class TwoDUnet():

    def __init__(self, model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        self.model = model


    def create_model(self, img_shape):

        concat_axis = 3

        inputs = layers.Input(shape=img_shape)
        conv1 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
        conv2 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
        maxpool1 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool1)
        conv4 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv3)
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool2)
        conv6 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv5)
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2))(conv6)

        conv7 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool3)
        conv8 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv7)
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2))(conv8)

        conv9 = layers.Conv2D(1024, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool4)
        conv10 = layers.Conv2D(1024, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)

        up_conv10 = layers.UpSampling2D(size=(2, 2))(conv10)
        up_samp1 = layers.concatenate([conv8, up_conv10], axis=concat_axis)
        conv11 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)

        conv12 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv11)
        conv13 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv12)

        up_conv13 = layers.UpSampling2D(size=(2, 2))(conv13)
        up_samp2 = layers.concatenate([conv6, up_conv13], axis=concat_axis)
        conv14 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp2)

        conv15 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv14)
        conv16 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv15)

        up_conv16 = layers.UpSampling2D(size=(2, 2))(conv16)
        up_samp3 = layers.concatenate([conv4, up_conv16], axis=concat_axis)
        conv17 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp3)

        conv18 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv17)
        conv19 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv18)

        up_conv19 = layers.UpSampling2D(size=(2, 2))(conv19)
        up_samp1 = layers.concatenate([conv2, up_conv19], axis=concat_axis)
        conv20 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)

        conv21 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv20)
        conv22 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv21)

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(conv22)

        model = models.Model(inputs=inputs, outputs=conv23)

        #model.compile(optimizer=SGD(lr=0.01, momentum=0.99, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef, binary_crossentropy])
        model.compile(optimizer=Adam(lr=0.001), loss=weighted_crossentropy, metrics=[dice_coef, binary_crossentropy, weighted_crossentropy])

        model.summary()

        return model


    def copy_crop(self, layer_target, layer_refer):

        # width, the 3rd dimension
        cw = (layer_target.get_shape()[2] - layer_refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 =  cw // 2, (cw // 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)
        # height, the 2nd dimension
        ch = (layer_target.get_shape()[1] - layer_refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)


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
                                       save_best_only=True,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=paths["log_path"],
                                           batch_size=batch_size,
                                           write_graph=False,
                                           write_grads=False,
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


    def predict_and_save(self, data, labels, output_path, batch_size=1):

        if not output_path.endswith('/'):
            output_path += '/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        predictions = self.model.predict(data, batch_size=batch_size, verbose=1)

        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):

            cv2.imwrite(output_path + 'original_' + str(index) + '.png', original)
            cv2.imwrite(output_path + 'prediction_' + str(index) + '.png', pred)
            cv2.imwrite(output_path + 'label_' + str(index) + '.png', label)


