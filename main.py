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


# Load params from auxiliar class
import model_parameters as mp
from torch.utils.tensorboard import SummaryWriter

model_class = mp.model_params()

# Run parameters
train_model = False
test_sample = True

print('semantic segmentation V1.0')

if train_model:
    trm.train_new_model(model_class, 3, 0.3, True, 35)

if test_sample:
    tsm.test_model(model_class)