import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from data_utils import dataset_util
from tqdm import tqdm
bar = tqdm(total=len(dataset_util.get_file_list('D:\herschel\\navigation\data\classification')))
for item in dataset_util.get_file_list('D:\herschel\\navigation\data\classification'):
    if 'jpg' in item:
        os.remove(item)
