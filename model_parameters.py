# This module class wraps all of the model parameters.

import os
import torch
import segmentation_models_pytorch as smp

class model_params:
    #Default inicialized values
    def __init__(self):
        #model_params
        self.encoder = 'se_resnext50_32x4d'
        self.classes = ['melanoma']
        self.pre_tr_weights = 'imagenet'
        self.device = 'cuda'
        self.act_func = 'sigmoid'

        #paths
        self.data = 'Data/'
        self.model_dir = 'Model/'
        self.file_name = 'se_resnext524d.pth'
        self.model_path = self.model_dir + self.file_name
        self.test_dir = 'Data/Test_images/'
        self.images_dir = os.path.join(self.data,"Training_input")
        self.masks_dir = os.path.join(self.data, "Training_annot")

        #model_config
        self.model = smp.FPN(encoder_name=self.encoder, encoder_weights=self.pre_tr_weights, classes=len(self.classes), activation=self.act_func)
        self.loss = smp.utils.losses.DiceLoss()
        self.metrics = [smp.utils.metrics.IoU(threshold=0.5),]
        self.optimizer = torch.optim.Adam([dict(params=self.model.parameters(),lr=0.0001),])

        #model_logs
        self.loss_logs = []
        self.score_logs = []
         
