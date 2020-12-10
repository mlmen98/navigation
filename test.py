import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from data_utils import dataset_util
from tqdm import tqdm

file_list = dataset_util.get_file_list('D:\herschel\\navigation\data\classification')
bar = tqdm(total=len(file_list))
for item in file_list:
    img = cv2.imread(item)
    img = cv2.resize(img, (512, 512))
    cv2.imwrite(item, img)
    bar.update()

