import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from data_utils import dataset_util
from tqdm import tqdm
import keras


def valid_loss_test(y_true, y_pred):
    y_true = keras.backend.reshape(y_true, (-1,))
    y_pred = keras.backend.reshape(y_pred, (-1, y_pred.shape[1].value))
    valid_weight = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), keras.backend.ones_like(y_true))
    y_true = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), y_true)
    loss = keras.backend.sum(valid_weight * keras.backend.sparse_categorical_crossentropy(y_true, y_pred))
    loss = loss / keras.backend.sum(valid_weight)
    return loss


if __name__ == '__main__':
    file_list = dataset_util.get_file_list('D:\herschel\\navigation\data\indoor_nav\\new\PixelLabelData\PixelLabelData')
    for item in file_list:
        img = cv2.imread(item, cv2.IMREAD_GRAYSCALE).astype(np.int16)
        img = img - 1
        img[img < 0] = 1
        img = 1 - img
        img = img*255
        cv2.imwrite(item, img.astype(np.uint8))

