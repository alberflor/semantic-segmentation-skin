import os
import torch
import numpy as np
import load_dataset as ld
import transformation as tfm
import test_model as tsm
import train_model as trm
import segmentation_models_pytorch as smp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Based on https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb tutorial

# Data paths.
DATA_DIR = "Data/"
model_dir = 'Model/'
file_name = 'se_resnext524d.pth'
model_path = model_dir + file_name

test_dir = 'Data/Test_images/'

#Model parameters.
ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['melanoma']
DEVICE = 'cuda'
ACTIVATION = 'sigmoid'

train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")
 
model = smp.FPN(encoder_name= ENCODER, encoder_weights=ENCODER_WEIGHTS, classes= len(CLASSES), activation= ACTIVATION)
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr= 0.0001),])


# Run parameters
train_model = False
test_sample = True

print('semantic segmentation V1.0')

if train_model:
    trm.train_new_model(model, train_img, train_mask, ENCODER, ENCODER_WEIGHTS, CLASSES, DEVICE, 12, 0.3, True, 42, loss, metrics, optimizer)

if test_sample:
    tsm.test_model(model_path, test_dir, ENCODER, ENCODER_WEIGHTS, CLASSES, DEVICE)