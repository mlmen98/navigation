import tensorflow as tf
import keras
from data_utils import dataset_util
import os
import random
import numpy as np
import cv2
import io
import PIL
from keras.callbacks import TensorBoard
from functools import partial
from keras_implementation import generator


def classification_branch(x):
    with tf.variable_scope('classification_branch'):
        classification_feature = x
        feature = keras.layers.Dropout(0.5)(classification_feature)
        output_p = keras.layers.Dense(5, activation='softmax', W_regularizer=keras.regularizers.l2(1e-3))(feature)
        return output_p
    

def atrous_spatial_pyramid_pooling_keras(inputs, output_stride=8, depth=256):
    """
    atrous spatial pyramid pooling implementation with keras
    :param inputs:
    :param output_stride:
    :param batch_norm_decay:
    :param is_training:
    :param depth:
    :return:
    """
    atrous_rates = [6, 12, 18]
    if output_stride == 8:
        atrous_rates = [2*item for item in atrous_rates]
    with tf.variable_scope('atrous_pyramid_pooling'):
        conv_1x1 = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(inputs)
        # conv_1x1 = tf.nn.batch_normalization(conv_1x1)
        conv_3x3_list = []
        for item in atrous_rates:
            conv_3x3 = keras.layers.Conv2D(depth, (3, 3), strides=1, dilation_rate=item, padding='same')(inputs)
            # conv_3x3 = tf.nn.batch_normalization(conv_3x3)
            conv_3x3_list.append(conv_3x3)
        with tf.variable_scope("image_level_features"):
            # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
            # 1×1 convolution with 256 filters( and batch normalization)
            image_level_features = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(image_level_features)
            # image_level_features = tf.nn.batch_normalization(image_level_features)
            # bilinearly upsample features
            inputs_size = tf.shape(inputs)[1:3]
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            net = tf.concat([conv_1x1]+conv_3x3_list+[image_level_features], axis=3, name='concat')
            net = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(net)
            # net = tf.nn.batch_normalization(net)
            return net


def bilinear(x, size=(512, 512)):
    x = tf.image.resize_bilinear(x, size)
    return x


def segmentation_branch(x, image_size=(512, 512)):
    """
    segmentation branch
    :param x:
    :return: a tensor
    """
    with tf.variable_scope('segmentation_branch'):
        x = atrous_spatial_pyramid_pooling_keras(x, 8, 256)
    # extract output tensor of block4
        with tf.variable_scope("upsampling_logits"):
                net = keras.layers.Conv2D(1, (1, 1), strides=1, padding='same', activation='sigmoid')(
                    x)
                net = keras.layers.Lambda(partial(bilinear, size=image_size))(net)
    return net


def valid_loss_for_classification(y_true, y_pred):
    """
    compute loss on valid input and ouput
    :param y_true:
    :param y_pre:
    :return:
    """
    y_true = keras.backend.reshape(y_true, (-1, ))
    y_pred = keras.backend.reshape(y_pred, (-1, 5))
    valid_weight = tf.where(tf.equal(y_true, 255), keras.backend.ones_like(y_true), keras.backend.zeros_like(y_true))
    y_true = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), y_true)
    loss = keras.backend.mean(valid_weight*keras.backend.sparse_categorical_crossentropy(y_true, y_pred))
    return loss


def valid_loss_for_segmentation(y_true, y_pred):
    y_true = keras.backend.reshape(y_true, (-1,))
    y_pred = keras.backend.reshape(y_pred, (-1,))
    valid_weight = tf.where(tf.equal(y_true, 255), keras.backend.ones_like(y_true), keras.backend.zeros_like(y_true))
    y_true = tf.where(tf.equal(y_true, 255), keras.backend.zeros_like(y_true), y_true)
    loss = keras.backend.mean(valid_weight * keras.backend.binary_crossentropy(y_true, y_pred))
    return loss



def model_fn():
    """
    return a keras model
    :return:
    """
    x = keras.Input((None, None, 3))
    backbone = keras.applications.ResNet50(input_tensor=x, include_top=False, pooling='avg', weights='imagenet')
    # free backbone
    for layer in backbone.layers:
        layer.trainable = False
    with tf.device('/gpu:0'):
        classification_feature = backbone.output
        classification_result = classification_branch(classification_feature)
    with tf.device('/gpu:1'):
        segmentation_feature = backbone.get_layer('activation_22').output # features from conv-3, downsample 3time * 256 filter
        segmentation_result = keras.layers.Lambda(segmentation_branch)(segmentation_feature)
    model = keras.models.Model(x, [classification_result, segmentation_result])
    print (model.summary())
    # metrics = ['acc']
    model.compile(
        loss=[valid_loss_for_classification, valid_loss_for_segmentation],
        optimizer='adam',
        # metrics=metrics
    )
    return model


def train():
    model = model_fn()
    # checkpoint = keras.callbacks.ModelCheckpoint(
    #     os.path.join(
    #         'classification_model',
    #         'classification'
    #     ),
    #     verbose=1
    # )
    train_generator = generator.combine_generator(folder='train', batch_size=16)
    # val_generator = generator.combine_generator(folder='val', batch_size=16)
    print ('start training...')
    model.fit_generator(generator=train_generator,
                         steps_per_epoch=1000,
                         epochs=50,
                         verbose=1,
                        # callbacks=[TensorBoard(log_dir='./log', histogram_freq=1, write_images=False,
                        #                            write_grads=1, write_graph=1)],
                        # validation_data=val_generator,
                        # validation_steps=1,
                        #  workers=3,
                         )
    model.save('classification')


if __name__ == '__main__':
    train()