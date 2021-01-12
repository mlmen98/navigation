from keras_implementation.visualization import apply_mask
import matplotlib.pyplot as plt
import numpy as np
import cv2


img_dir = 'D:\herschel\\navigation\\test\data\\40.jpg'
label_dir = 'D:\herschel\\navigation\\test\data\\40_mask.png'
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
mask = cv2.imread(label_dir, 0)
mask[mask > 0] = 1
color = [0, 75, 75]
img = apply_mask(img, np.squeeze(mask), color)
img = np.uint8(img)
plt.imsave('test.png', img)