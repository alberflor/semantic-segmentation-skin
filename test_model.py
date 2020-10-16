import torch
import load_dataset as ds
import transformation as tfm
import segmentation_models_pytorch as smp
import numpy as np
import cv2

model_dir = 'Model/'
#file_name = 'se_resnext524d.pth'
file_name = 'best_model.pth'

model_path = model_dir + file_name
test_dir = 'Data/Test_images/'
save_path = 'Data/Predicted_masks/'

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['melanoma']
DEVICE = 'cuda'


def test_model(m_path, t_path, s_path, encoder, weights, classes, device):
    # Load model
    model = torch.load(m_path)
    prep_fn = smp.encoders.get_preprocessing_fn(encoder, weights)

    # Load data to test
    vis_dataset = ds.testing_data(t_path,
    classes,
    augmentation=tfm.get_validation_augmentation())

    test_dataset = ds.testing_data(t_path,
    classes, 
    augmentation=tfm.get_validation_augmentation(), 
    preprocessing=tfm.get_preprocessing(prep_fn))

    # Select random sample
    n = np.random.choice(len(vis_dataset))
    vis = vis_dataset[n].astype('uint8')
    test_image = test_dataset[n]

    # Predict using model
    x_tensor = torch.from_numpy(test_image).to(device).unsqueeze(0)
    pred_mask = model.predict(x_tensor)
    pred_mask = (pred_mask.squeeze().cpu().numpy().round())
    cv2.imwrite(s_path)

    #Visualize prediction
    ds.visualize(image=vis, predicted=pred_mask)

def test_wrap(a):
    print(a.model_params.model_path)

import model_parameters as mp

print(mp.model_params.model_path)
test_wrap(mp.model_params.model_path)
