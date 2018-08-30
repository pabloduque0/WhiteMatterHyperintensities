from keras import backend as K
import tensorflow as tf
import numpy as np


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_pred_f * y_true_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_crossentropy_pixelwise(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 10e-8, 1. - 10e-8)
    wmh_indexes = np.where(y_true == 1.0)
    weights = np.repeat(0.3, 240*240)
    weights = np.reshape(weights, (1, 240, 240, 1))
    weights[wmh_indexes] = 0.7
    crossentropy = -np.sum(y_true * weights * K.log(y_pred))

    return crossentropy

def sigmoid(x):
    return 1. / (1. + K.exp(-x))

def weighted_crossentropy_pix(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_pred = K.log(y_pred / (1 - y_pred))

    wmh_indexes = np.where(y_true == 1.0)
    weights = np.repeat(0.3, 240 * 240)
    weights = np.reshape(weights, (1, 240, 240, 1))
    weights[wmh_indexes] = 2.0

    crossentropy = (y_true * weights * -K.log(sigmoid(y_pred)) + (1 - y_true) * -K.log(1 - sigmoid(y_pred)))
    return crossentropy



dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

weighted_crossentropy = weighted_crossentropy_pix