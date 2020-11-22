import torch
import data
import transformation as tfm
import load_dataset as ld
from torch.utils.data.sampler import SubsetRandomSampler
import segmentation_models_pytorch as smp
import numpy as np
import os
import matplotlib.pyplot as plt



def train_new_model(m, batch, val_size, shuffle, seed_num):

    preprocessing_fn = smp.encoders.get_preprocessing_fn(m.encoder, m.pre_tr_weights)
    # Loading data.
    dataset = ld.Dataset(m.images_dir,
    m.masks_dir,
    classes= m.classes,
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
    train_epoch = smp.utils.train.TrainEpoch(m.model,
    loss=m.loss,
    metrics=m.metrics,
    optimizer=m.optimizer,
    device=m.device,
    verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
    m.model,
    loss=m.loss,
    metrics=m.metrics,
    device=m.device,
    verbose=True,
    )

    max_score = 0
    train_epoch_logs = []
    valid_epoch_logs = []

    for i in range(0,20):
        print('\n Epoch: {}'.format(i))
        train_logs, loss_t, metric_t = train_epoch.run(train_loader)
        
        loss_array = np.asarray(loss_t)
        loss_values = [k['dice_loss'] for k in loss_array]
        m.loss_logs = loss_values

        #Plot dice loss
        data.plot_log(loss_values, 0.8, "Coeficiente de dados", "Iteración", False, False)

        # Save iteration logs
        data.save_df(loss_values, "train_it_ep_"+str(i), True)

        #Append epoch logs.
        train_epoch_logs.append(train_logs)
        
        valid_logs, loss_v, metric_v = valid_epoch.run(valid_loader)
        
        metric_array = np.asarray(metric_v)
        metric_values = [k['iou_score'] for k in metric_array]
        m.score_logs = metric_values

        #Plot IoU score
        data.plot_log(metric_values, 1, "Criterio de Jaccard", "Iteración", False, False)

        # Save iteration logs.
        data.save_df(metric_values, "valid_it_ep_"+str(i), True)

        #Append epoch logs.
        valid_epoch_logs.append(metric_v)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(m.model,('Model/'+ m.encoder + '.pth'))
            print('Highest Score Model Saved: {}'.format(max_score))

        if i == 10:
            m.optimizer.param_groups[0]['lr'] = 1e-5
            print('decreased decoder learning rate to 1e-5')

    # Save data logs / epochs.
    
    data.save_df(train_epoch_logs, "train_resumed", True)
    data.save_df(valid_epoch_logs, "validation_resumed", True)
    
 