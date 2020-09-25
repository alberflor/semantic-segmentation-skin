import torch
import transformation as tfm
import load_dataset as ld
from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp
import numpy as np
import os

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DATA_DIR = "Data/"
CLASSES = ['melanoma']
DEVICE = 'cuda'
ACTIVATION = 'sigmoid'
train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

#Model parameters. 
model = smp.FPN(encoder_name= ENCODER, encoder_weights=ENCODER_WEIGHTS, classes= len(CLASSES), activation= ACTIVATION)
loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr= 0.0001),])

def train_new_model(model, images, masks, encoder, weights, class_arr, device,  batch, val_size, shuffle, seed_num, loss, metrics, optimizer):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, weights)
    # Loading data.
    dataset = ld.Dataset(images,
    masks,
    classes= class_arr,
    augmentation=tfm.get_training_augmentation(),
    preprocessing=tfm.get_preprocessing(preprocessing_fn))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_size * dataset_size))
    
    if shuffle:
        np.random.seed(seed_num)
        np.random.shuffle(indices)
    
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, sampler=valid_sampler)

    # Epoch configuration.

    train_epoch = smp.utils.train.TrainEpoch(model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=device,
    verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=device,
    verbose=True,
    )

    max_score = 0 
    for i in range(0,40):
        print('\n Epoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, 'Model/best_model.pth' )
            print('Best Model Saved')
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('decreased decoder learning rate to 1e-5')

# Test module
#train_new_model(model, train_img, train_mask, ENCODER, ENCODER_WEIGHTS, CLASSES, DEVICE, 12, 0.3, True, 42, loss, metrics, optimizer)