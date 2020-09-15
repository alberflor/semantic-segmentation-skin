import os
import torch
import numpy as np
import load_dataset
import transformation
import segmentation_models_pytorch as smp
from torch.utils.data.sampler import SubsetRandomSampler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Based on https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb tutorial

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
DATA_DIR = "Data/"
CLASSES = ['melanoma']
DEVICE = 'cuda'
ACTIVATION = 'sigmoid'

train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

#Model parameters.
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)  

model = smp.FPN(encoder_name= ENCODER, encoder_weights=ENCODER_WEIGHTS, classes= len(CLASSES), activation= ACTIVATION)

loss = smp.utils.losses.DiceLoss()
metrics = [smp.utils.metrics.IoU(threshold=0.5),]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr= 0.0001),])

# Dataset loading and transformations (.resize + .to_tensor)

dataset = load_dataset.Dataset(train_img, train_mask, 
    augmentation=transformation.get_training_augmentation(),
    preprocessing=transformation.get_preprocessing(preprocessing_fn),
    classes=['melanoma'],
    )

# Split into training and validation subsets.
batch_size = 12
validation_size = 0.3
shuffle_data = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_size * dataset_size))

if shuffle_data:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)


#Model training 

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss = loss,
    metrics = metrics,
    optimizer= optimizer,
    device= DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss = loss,
    metrics = metrics,
    device= DEVICE,
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