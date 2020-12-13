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


def generator(data_folder=None, batch_size=16):
    class_list = ['turn_left', ' turn_right', 'move_forward', 'turn_around', 'target_found']
    if data_folder is None:
        data_folder = 'D:\herschel\\navigation\data\classification\\train'
    samples = []
    labels = []
    for item in class_list:
        path = os.path.join(data_folder, item)
        item_list = (dataset_util.get_file_list(path))
        samples += item_list
        labels += [class_list.index(item)]*len(item_list)
    samples = np.array(samples)
    labels = np.array(labels)
    shuffle_index = np.arange(0, len(samples))
    np.random.shuffle(shuffle_index)
    samples = (samples[shuffle_index])
    labels = list(labels[shuffle_index])
    while True:
        batch_index = np.random.randint(0, len(samples), batch_size)
        batch_data_list = list(samples[batch_index])
        batch_data = []
        batch_label = [labels[item] for item in batch_index]
        for path_item in batch_data_list:
            with tf.gfile.GFile(path_item, 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = PIL.Image.open(encoded_jpg_io)
            batch_data.append(np.array(image))
        batch_data = np.stack(batch_data, axis=0)
        batch_label = np.array(batch_label)
        yield batch_data, batch_label


def model_fn(x):
    backbone = keras.applications.ResNet50(input_tensor=x, include_top=False, pooling='avg', weights='imagenet')
    # free backbone
    for layer in backbone.layers:
        layer.trainable = False
    feature = backbone.output
    feature = keras.layers.Dropout(0.5)(feature)
    output_p = keras.layers.Dense(5, activation='softmax', W_regularizer=keras.regularizers.l2(1e-3))(feature)
    return output_p


def train():
    x = keras.Input((None, None, 3))
    output = model_fn(x)
    model = keras.Model(x, output)
    # opt = keras.optimizers.SGD(lr=1e-6, momentum=0.9, decay=0.0, nesterov=False)
    metrics = ['acc']
    loss = keras.losses.sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            'classification_model',
            'classification'
        ),
        verbose=1
    )
    model.fit_generator(generator=generator(),
                         steps_per_epoch=100,
                         epochs=5,
                         verbose=1,
                        # callbacks=[TensorBoard(log_dir='./classification_model', histogram_freq=1, write_images=True,
                        #                            write_grads=1, write_graph=1), checkpoint],
                        validation_data=partial(generator, data_folder='D:\herschel\\navigation\data\classification\\validation', batch_size=16)(),
                        validation_steps=10,
                         workers=3,
                         )
    model.save('classification')


if __name__ == '__main__':
    train()