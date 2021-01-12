from postprocessing import utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology


def mask_refine_entry(label, motor_command=None):
    """
    refine mask from segmentation result
    :param img: 3d
    :param label: 1d of (0, 1)
    :param motor_command: motor_command from motor prediction branch
    :return:
    """
    # find the contous with largest area
    _, contours, _ = cv2.findContours(label.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)[-1]
    label = np.zeros_like(label, dtype=np.uint8)
    cv2.drawContours(label, [contours], -1, (1,), -1)
    # fill small holes
    label = morphology.remove_small_holes(label, 64, in_place=True).astype(np.uint8)
    return label


def good_feature_point_to_track(img, label):
    mask = np.stack([label]*3, axis=2)
    img_mask = img*mask
    # img_mask = img
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    feature_point_mask = np.zeros_like(img)
    feature_point_mask = cv2.drawKeypoints(feature_point_mask, kp, feature_point_mask, color=(0, 255, 0), flags=0)
    feature_point_mask = feature_point_mask*mask
    draw = feature_point_mask + img
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    plt.imshow(draw), plt.show()
    return 0


def test():
    img_dir = 'D:\herschel\\navigation\postprocessing\samples_before\\test_full.jpg'
    img = cv2.imread(img_dir)
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    draw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(draw), plt.show()


if __name__ == '__main__':
    import os
    img_list = os.listdir('D:\herschel\\navigation\postprocessing\samples_before')
    img_list = [os.path.join('D:\herschel\\navigation\postprocessing\samples_before', item) for item in img_list]
    img = cv2.imread(img_list[2])
    img, label = utils.split(img)
    label = mask_refine_entry(label)
    plt.imshow(label)
    plt.show()
    good_feature_point_to_track(img, label)
    # test()