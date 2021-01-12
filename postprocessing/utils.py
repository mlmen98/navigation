"""
utils function for postprocessing
"""
import cv2
import numpy as np


def split(img_data):
    """
    split img_data hstack of [img, label] into img and label
    :param img_data: 3D numpy array for a hstack[img, label]
    :return: img , label
    """
    img, label = np.hsplit(img_data, 2)
    label = label[:, :, 0] / 255
    label = np.uint8(label)
    return img, label