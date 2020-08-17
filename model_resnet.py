import torch
import numpy as np
import segmentation_models_pytorch as smp

#Creation of resnet model.

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['melanoma']
ACTIVACION = ['relu']
DEVICE = 'cuda'

model = smp.FPN(
    encoder_name = ENCODER,
    encoder_weights= ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVACION,
)

train_dataset = Dataset(
    x_train_dir, 
    y_train_dir, 
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

valid_dataset = Dataset(
    x_valid_dir, 
    y_valid_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

