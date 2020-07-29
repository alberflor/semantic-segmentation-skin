import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = "Data/"

train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

def visualize_matrix(dataset, v):

    fig = plt.figure(figsize=(12, 6))
    for i in range(v):
        image, mask = dataset[i]
        img_dict = {}
        img_dict['image'] = image
        img_dict['mask'] = mask
        n = len(img_dict)

        for j,(name, image) in enumerate(img_dict.items()):
          plt.subplot(v, n , i + j + 1)
          plt.imshow(image)
    plt.show()

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, ))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['melanoma']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        # Path to images.    
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # Path to segmentation masks.
        self.masks_fps = [os.path.join(masks_dir, (image_id[:-4]+'_segmentation.png')) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        print('Id: ',i)
        print('Img file: ',self.images_fps[i])
        print('Img mask: ', self.masks_fps[i])
        print('Img shape: ', image.shape)
        print('Mask Shape: ', mask.shape)

        # extract certain classes from mask
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

dataset = Dataset(train_img, train_mask, classes=['melanoma'])
image , mask = dataset[3]

visualize(image = image, mask = mask.squeeze())