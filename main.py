import os
import load_dataset
import torch
import numpy as np
import segmentation_models_pytorch as smp

# Based on https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb tutorial

CLASSES = ['melanoma']
DATA_DIR = "Data/"

train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

dataset = load_dataset.Dataset(train_img, train_mask, classes=['melanoma'])
image, mask = dataset[5]

load_dataset.visualize(image= image, mask = mask.squeeze())

#Model creation
model = smp.FPN(encoder_name='se_resnet50_32x4d', encoder_weights='imagenet', classes= 1, activation= 'relu')
preprocessing_fn = smp.encoders.get_preprocessing_fn('se_resnet50_32x4d', 'imagenet')

#Training
training_dataset = load_dataset.Dataset(train_img, train_mask, classes=CLASSES)
#validation_dataset = load_dataset.Dataset(train_img, train_mask, classes=CLASSES)
