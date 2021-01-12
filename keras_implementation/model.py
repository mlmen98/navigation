"""
model building with keras api
"""
import tensorflow as tf
import keras
import numpy as np
from keras_implementation.config import *
from keras_implementation.train_utils import valid_loss_test, my_iou_metric


def classification_branch(x):
    with tf.variable_scope('classification_branch'):
        classification_feature = x
        if ModelConfig.backbone == 'VGG':
            classification_feature = keras.layers.AveragePooling2D((8, 8))(classification_feature)
            classification_feature = keras.layers.Flatten()(classification_feature)
        feature = keras.layers.Dropout(0.5)(classification_feature)
        output_p = keras.layers.Dense(7, activation='softmax', W_regularizer=keras.regularizers.l2(1e-2), name='classification')(feature)
        return output_p
    

def atrous_spatial_pyramid_pooling_keras(inputs, output_stride=8, depth=32):
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
        conv_1x1 = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same', activation=None, kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
        conv_1x1 = keras.layers.BatchNormalization()(conv_1x1)
        conv_1x1 = keras.layers.Activation('relu')(conv_1x1)
        conv_3x3_list = []
        for item in atrous_rates:
            conv_3x3 = keras.layers.Conv2D(depth, (3, 3), strides=1, dilation_rate=item, padding='same', activation=None, kernel_regularizer=keras.regularizers.l2(0.01))(inputs)
            conv_3x3 = keras.layers.BatchNormalization()(conv_3x3)
            conv_3x3 = keras.layers.Activation('relu')(conv_3x3)
            conv_3x3_list.append(conv_3x3)
        # with tf.variable_scope("image_level_features"):
        #     # global average pooling
        #     image_level_features = keras.layers.Lambda(reduce_mean)(inputs)
        #     image_level_features = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same', activation=None)(image_level_features)
        #     image_level_features = keras.layers.BatchNormalization()(image_level_features)
        #     image_level_features = keras.layers.Activation('relu')(image_level_features)
        #     # bilinearly upsample features
        #     image_level_features = Bilinear([image_level_features, inputs])
        with tf.variable_scope('pyramid_concat'):
                net = keras.layers.Concatenate(axis=3, name='concat')([conv_1x1]+conv_3x3_list)#+[image_level_features])
                net = keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same', activation=None, kernel_regularizer=keras.regularizers.l2(0.01))(net)
                net = keras.layers.BatchNormalization()(net)
                net = keras.layers.Activation('relu')(net)
                return net


def bilinear(tensor):
    new_shape = tf.shape(tensor)[1:3]
    new_shape *= tf.constant(np.array([1, 1]).astype('int32'))
    tensor = tf.image.resize_bilinear(tensor, new_shape)
    return tensor


def reduce_mean(x, axis=(1, 2), keepdims=True):
    return tf.reduce_mean(x, axis=axis, keepdims=keepdims)


def bilinear_1(input_tensor):
    return tf.image.resize_bilinear(input_tensor, (512, 512))


def Bilinear(input_tensor, shape=(512, 512)):
    def bilinear(tensor):
        tensor = tf.image.resize_bilinear(tensor, shape)
        return tensor
    return keras.layers.Lambda(bilinear)(input_tensor)


def segmentation_branch(x):
    """
    segmentation branch
    :param x:
    :return: a tensor
    """
    with tf.variable_scope('segmentation_branch'):
        x = atrous_spatial_pyramid_pooling_keras(x, 8, 32)
    # extract output tensor of block4
        with tf.variable_scope("upsampling_logits"):
                x = keras.layers.Conv2D(2, (1, 1), strides=1, padding='same', activation=None, kernel_regularizer=keras.regularizers.l2(0.01))(x)
                x = keras.layers.Lambda(bilinear_1)(x)
                x = keras.layers.Softmax(axis=-1, name='segmentation')(x)
    return x


def model_fn():
    """
    build the unit model
    :return:
    """
    x = keras.Input((512, 512, 3))
    if ModelConfig.backbone == 'ResNet':
        backbone = keras.applications.ResNet50(input_tensor=x, include_top=False, pooling='avg', weights='imagenet')
    else:
        backbone = keras.applications.VGG16(input_tensor=x, include_top=False, pooling=None, weights='imagenet')
    # free backbone
    for layer in backbone.layers:
        layer.trainable = False
        if ModelConfig.backbone == 'VGG':
            if 'block5' in layer.name:
                layer.trainable = ModelConfig.backbone_trainable
        if ModelConfig.backbone == 'ResNet':
            if 'res5' in layer.name:
                layer.trainable = ModelConfig.backbone_trainable
    with tf.device('/gpu:0'):
        classification_feature = backbone.output
        classification_result = classification_branch(classification_feature)
    with tf.device('/gpu:1'):
        if ModelConfig.backbone == 'ResNet':
            segmentation_feature = backbone.get_layer('activation_22').output # P3-512
        else:
            segmentation_feature = backbone.get_layer('block4_conv3').output # P3-512
        segmentation_result = segmentation_branch(segmentation_feature)
    # summary
    tf.summary.image('features', x, max_outputs=6)  # Concatenate row-wise.
    tf.summary.image('Pre_label', tf.concat([255 * tf.cast(segmentation_result, tf.float32)] * 3, axis=3),
                     max_outputs=6)  # Concatenate row-wise.
    # metrics = [valid_acc_metric]
    metrics = ['acc']
    classification_model = keras.models.Model(x, classification_result)
    classification_model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer='adam', #keras.optimizers.SGD(lr=3e-5, momentum=0.9),
        metrics=['acc']
    )
    segmentation_model = keras.models.Model(x, segmentation_result)
    segmentation_model.compile(
        loss=keras.losses.sparse_categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=1e-3, momentum=0.9),
        metrics=['acc', my_iou_metric]
    )
    unite_model = keras.models.Model(x, [classification_result, segmentation_result])
    unite_model.compile(
        loss=[valid_loss_test, valid_loss_test],
        optimizer='adam',
        metrics=metrics
    )
    return unite_model, classification_model, segmentation_model


if __name__ == '__main__':
    # plot model
    u, c, s = model_fn()
    from keras.utils import plot_model
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    plot_model(u, show_shapes=True, to_file='united_model.png')
    plot_model(c, show_shapes=True, to_file='classification_model.png')
    plot_model(s, show_shapes=True, to_file='segmentation_model.png')