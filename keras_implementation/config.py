"""
config for training and model
"""


class ModelConfig(object):
    backbone = ['VGG', 'ResNet'][1]
    # unfreeze conv layers to P5 feature level (5mean downsample 5 time to the input image)
    backbone_trainable = False


class TrainingConfig(object):
    training_branch = ['unit', 'segmentation', 'classification'][1]
    # training segmentation branch with cityscape data set instead of collected indoor dataset
    segmentation_on_cityscape = False


# prepare gpu for training
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
if TrainingConfig.training_branch == 'segmentation':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif TrainingConfig.training_branch == 'classification':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"