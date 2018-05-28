from keras import models
from keras import layers

class ThreeDUnet():

    def __init__(self):
        self.model = None

    def create_model(self, img_shape):

        concat_axis = 4

        inputs = layers.Input(shape=img_shape)

        conv1 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(inputs)
        conv2 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(conv1)
        maxpool1 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv2)

        conv3 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(maxpool1)
        conv4 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(conv3)
        maxpool2 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv4)

        conv5 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(maxpool2)
        conv6 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv5)
        maxpool3 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv6)

        conv7 = layers.Conv3D(512, kernel_size=3, padding='same', activation='relu')(maxpool3)
        conv8 = layers.Conv3D(512, kernel_size=3, padding='same', activation='relu')(conv7)
        maxpool4 = layers.MaxPool3D(pool_size=(2, 2, 2))(conv8)

        conv9 = layers.Conv3D(1024, kernel_size=3, padding='same', activation='relu')(maxpool4)
        conv10 = layers.Conv3D(1024, kernel_size=3, padding='same', activation='relu')(conv9)

        up_conv10 = layers.UpSampling3D(size=(2, 2, 2))(conv10)
        ch, cw, cs = self.copy_crop(conv8, up_conv10)
        crop_conv8 = layers.Cropping3D(cropping=(ch, cw, cs))(conv8)
        up_samp1 = layers.concatenate([up_conv10, crop_conv8], axis=concat_axis)
        conv11 = layers.Conv3D(512, kernel_size=2, padding='same', activation='relu')(up_samp1)

        conv12 = layers.Conv3D(512, kernel_size=3, padding='same', activation='relu')(conv11)
        conv13 = layers.Conv3D(512, kernel_size=3, padding='same', activation='relu')(conv12)

        up_conv13 = layers.UpSampling3D(size=(2, 2, 2))(conv13)
        ch, cw, cs = self.copy_crop(conv6, up_conv13)
        crop_conv6 = layers.Cropping3D(cropping=(ch, cw, cs))(conv6)
        up_samp2 = layers.concatenate([up_conv13, crop_conv6], axis=concat_axis)
        conv14 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(up_samp2)

        conv15 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv14)
        conv16 = layers.Conv3D(256, kernel_size=3, padding='same', activation='relu')(conv15)

        up_conv16 = layers.UpSampling3D(size=(2, 2, 2))(conv16)
        ch, cw, cs = self.copy_crop(conv4, up_conv16)
        crop_conv4 = layers.Cropping3D(cropping=(ch, cw, cs))(conv4)
        up_samp3 = layers.concatenate([up_conv16, crop_conv4], axis=concat_axis)
        conv17 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(up_samp3)

        conv18 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(conv17)
        conv19 = layers.Conv3D(128, kernel_size=3, padding='same', activation='relu')(conv18)

        up_conv19 = layers.UpSampling3D(size=(2, 2, 2))(conv19)
        ch, cw, cs = self.copy_crop(conv2, up_conv19)
        crop_conv2 = layers.Cropping3D(cropping=(ch, cw, cs))(conv2)
        up_samp1 = layers.concatenate([up_conv19, crop_conv2], axis=concat_axis)
        conv20 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(up_samp1)

        conv21 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(conv20)
        conv22 = layers.Conv3D(64, kernel_size=3, padding='same', activation='relu')(conv21)

        conv23 = layers.Conv3D(2, kernel_size=1, padding='same', activation='relu')(conv22)

        model = models.Model(inputs=inputs, outputs=conv23)

        self.model = model


    def copy_crop(self, layer_target, layer_refer):

        cs = (layer_target.get_shape()[3] - layer_refer.get_shape()[3]).value
        assert (cs >= 0)
        if cs % 2 != 0:
            cs1, cs2 = cs // 2, (cs // 2) + 1
        else:
            cs1, cs2 = cs // 2, (cs // 2)
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

        return (ch1, ch2), (cw1, cw2), (cs1, cs2)


