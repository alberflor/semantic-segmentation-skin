import os
import load_dataset
import torch
import numpy as np
import segmentation_models_pytorch as smp

# Based on https://github.com/qubvel/segmentation_models.pytorch tutorial

DATA_DIR = "Data/"
train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

dataset = load_dataset.Dataset(train_img, train_mask, classes=['melanoma'])
image, mask = dataset[5]

load_dataset.visualize(image= image, mask = mask.squeeze())

model = smp.FPN('resnet34', in_channels=1)
