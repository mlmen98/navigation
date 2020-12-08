import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

with tf.Session() as sess:
    labels = tf.constant([0, 0, 1], dtype=tf.int32)
    predict_p = np.array([6, 4, 99, 1, 5, 5]).reshape(3, 2)
    print (predict_p)
    predict_p = tf.constant(predict_p, dtype=tf.float32)
    p = tf.nn.softmax(predict_p)
    segmentation_loss = (tf.keras.losses.sparse_categorical_crossentropy(labels, p))
    s = tf.losses.sparse_softmax_cross_entropy(labels, predict_p)
    print (sess.run(segmentation_loss))
    # print (sess.run(p))
    print (sess.run(s))
