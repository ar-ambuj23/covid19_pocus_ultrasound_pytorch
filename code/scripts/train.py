#!/usr/bin/env python
# coding: utf-8

# Import required libraries

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('../models')
from vgg import VGG16_model

import os
import random
from imutils import paths
from collections import defaultdict 
import numpy as np
import time

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# ### Fixing Random Seeds
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data_dir', typr=str, default='../data/cross_validation', help='path to input cross-validation dataset')
ap.add_argument('-m', '--model_name', type=str, default='vgg16')
ap.add_argument('-s', '--model_save_dir', type=str, default='../trained_models')
ap.add_argument('-f', '--fold', type=int, default='0', help='fold to take as test data')
ap.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
ap.add_argument('-ep', '--epochs', type=int, default=20)
ap.add_argument('-bs', '--batch_size', type=int, default=16)
ap.add_argument('-iw', '--img_width', type=int, default=224)
ap.add_argument('-ih', '--img_height', type=int, default=224)
args = vars(ap.parse_args())


# ### Initializing parameters
CROSS_VAL_DIR = args['data_dir']
MODEL_NAME = args['model_name']
MODEL_SAVE_DIR = args['model_save_dir']
FOLD = args['fold']
LR = args['learning_rate']
N_EPOCHS = args['epochs']
BATCH_SIZE = args['batch_size']
IMG_WIDTH = args['img_width']
IMG_HEIGHT = args['img_height']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")