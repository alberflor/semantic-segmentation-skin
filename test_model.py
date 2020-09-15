import torch
import load_dataset as ds
import transformation as tfm
import segmentation_models_pytorch as smp
import numpy as np

model_dir = 'Model/'
file_name = 'se_resnext524d.pth'
test_dir = 'Data/Test_images/'

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'

CLASSES = ['melanoma']
DEVICE = 'cuda'

# Load model.

test_model = torch.load(model_dir+file_name)

# Load test image.

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)  

vis_dataset = ds.testing_data(test_dir=test_dir,classes=CLASSES,)
test_dataset = ds.testing_data(test_dir=test_dir,classes=CLASSES,augmentation=tfm.get_validation_augmentation(), preprocessing=tfm.get_preprocessing(preprocessing_fn))

n = np.random.choice(len(vis_dataset))
vis = vis_dataset[n].astype('uint8')

test_image = test_dataset[n]

#ds.visualize(image=image_test)

# Test model.
x_tensor = torch.from_numpy(test_image).to(DEVICE).unsqueeze(0)
pr_mask = test_model.predict(x_tensor)
pr_mask = (pr_mask.squeeze().cpu().numpy().round())

# Visualize mask
ds.visualize(image=vis, predicted=pr_mask)