"""
util functions and layers for buiding model
"""
import tensorflow as tf

from config import ModelConfig


def atrous_spatial_pyramid_pooling_keras(inputs, output_stride, depth=256):
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
        conv_1x1 = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(inputs)
        # conv_1x1 = tf.nn.batch_normalization(conv_1x1)
        conv_3x3_list = []
        for item in atrous_rates:
            conv_3x3 = tf.keras.layers.Conv2D(depth, (3, 3), strides=1, dilation_rate=item, padding='same')(inputs)
            # conv_3x3 = tf.nn.batch_normalization(conv_3x3)
            conv_3x3_list.append(conv_3x3)
        with tf.variable_scope("image_level_features"):
            # global average pooling
            image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
            # 1×1 convolution with 256 filters( and batch normalization)
            image_level_features = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(image_level_features)
            # image_level_features = tf.nn.batch_normalization(image_level_features)
            # bilinearly upsample features
            inputs_size = tf.shape(inputs)[1:3]
            image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')
            net = tf.concat([conv_1x1]+conv_3x3_list+[image_level_features], axis=3, name='concat')
            net = tf.keras.layers.Conv2D(depth, (1, 1), strides=1, padding='same')(net)
            # net = tf.nn.batch_normalization(net)
            return net


def classification_branch(x):
    """
    classification branch
    :param x: input tensor
    :return: a tensor
    """
    with tf.variable_scope('classification_branch'):
        tf.summary.histogram("feature_for_classification", x)
        x = tf.nn.dropout(x, 0.5)
        x = tf.keras.layers.Dense(ModelConfig.classification_num_classes, activation='softmax')(x)
        tf.summary.histogram("classification_softmax_output", x)
    return x


def segmentation_branch(x, image_size):
    """
    segmentation branch
    :param x:
    :return: a tensor
    """
    with tf.variable_scope('segmentation_branch'):
        x = atrous_spatial_pyramid_pooling_keras(x, 8, 256)
    # extract output tensor of block4
        with tf.variable_scope("upsampling_logits"):
                net = tf.keras.layers.Conv2D(ModelConfig.seg_num_classes, (1, 1), strides=1, padding='same', activation=None)(
                    x)
                logits = tf.image.resize_bilinear(net, image_size, name='upsample')
                segmentation_result = tf.nn.softmax(logits, name='softmax_tensor')
    return segmentation_result


def split_vgg_var():
    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ResNet')
    head_var = all_var[:144]
    tail_var = all_var[144:]
    return head_var, tail_var


def drop_bn_weight(var_list):
    """
    drop BN DELTA/GAMMA weight, thus disable BN
    :param var_list:
    :return:
    """
    var_list = [v for v in var_list
                      if 'beta' not in v.name and 'gamma' not in v.name]
    return var_list


def get_train_var(free_vgg, bn_trainable):
    """
    get train var list to two branch
    :param free_vgg:
    :param bn_trainable:
    :return:
    """
    vgg_head_var, vgg_tail_var = split_vgg_var()
    classification_train_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'classification_branch')
    segmentation_train_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'segmentation_branch')
    if not free_vgg:
        classification_train_var += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'ResNet')
        segmentation_train_var += vgg_head_var
    if not bn_trainable:
        classification_train_var = drop_bn_weight(classification_train_var)
        segmentation_train_var = drop_bn_weight(segmentation_train_var)
    return classification_train_var, segmentation_train_var