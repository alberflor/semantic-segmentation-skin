import torch
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
    for i in range(0,40):
        print('\n Epoch: {}'.format(i))
        train_logs, loss_t, metric_t = train_epoch.run(train_loader)
        
        loss_array = np.asarray(loss_t)
        loss_values = [k['dice_loss'] for k in loss_array]
        print(loss_values)

        m.loss_logs = loss_values
        fig_1 = plt.figure()
        plt.plot(loss_values)
        plt.ylabel('Coeficiente de dados', fontsize=12)
        plt.xlabel('Iteración', fontsize=12)
        plt.savefig('Plots/'+'dl_epoch_'+str(i))
        
        
        valid_logs, loss_v, metric_v = valid_epoch.run(valid_loader)
        
        metric_array = np.asarray(metric_v)
        metric_values = [k['iou_score'] for k in metric_array]
        print(metric_values)

        m.score_logs = metric_values

        fig_2 = plt.figure()
        plt.plot(metric_values)
        plt.ylabel('Índice de Jaccard', fontsize=12)
        plt.xlabel('Iteración', fontsize=12)
        plt.savefig('Plots/'+'score_epoch_'+str(i))
        
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(m.model,('Model/'+ m.encoder + '.pth'))
            print('Highest Score Model Saved: {}'.format(max_score))
        if i == 25:
            m.optimizer.param_groups[0]['lr'] = 1e-5
            print('decreased decoder learning rate to 1e-5')
 