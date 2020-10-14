# This module class wraps all of the model parameters.

import os
import segmentation_models_pytorch as smp
class model_params:

    #model_params
    encoder = 'se_resnext50_32x4d'
    classes = ['melanoma']
    pre_tr_weights = 'imagenet'
    device = 'cuda'
    act_func = 'sigmoid'

    #paths
    data = 'Data/'
    model_dir = 'Model/'
    file_name = 'se_resnext524d.pth'
    model_path = model_dir + filename
    test_dir = 'Data/Test_images'

    images_dir = os.path.join(data,"Training_input")
    masks_dir = os.path.join(data, "Training_annot")

    model = smp.FPN(encoder_name=encoder, encoder_weights=pre_tr_weights, classes=len(classes), activation=act_func)
    loss = smp.utils.losses.DiceLoss()
    

    #run_params
    train_new_model = False
    test_random_sample = True
    