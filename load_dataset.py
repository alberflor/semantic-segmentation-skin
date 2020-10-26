import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = "Data/"
train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

def visualize(**images):
    """Plot images in one row."""
    n = len(images)
    plt.figure(figsize=(10, 4))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
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
        # Path to images and filter placeholder.    
        self.ids = os.listdir(images_dir)

        #Split dataset, let empty for full usage.
        self.ids = self.ids[:300]

        self.filtered = [file for file in self.ids if file.endswith(".placeholder")]

        # Remove filtered
        for file in self.filtered:
            self.ids.remove(file)

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]

        # Path to segmentation masks.
        self.masks_fps = [os.path.join(masks_dir, (image_id[:-4]+'_segmentation.png')) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):

        # read data and apply transformations.
        try:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        except AssertionError:
            print('Assertion error')
            
        # read masks and print logs.
        mask = cv2.imread(self.masks_fps[i], 0)
    
        #print('Id: ',i)
        #print('Img file: ',self.images_fps[i])
        #print('Img mask: ', self.masks_fps[i])
        #print('Img shape: ', image.shape)
        #print('Mask Shape: ', mask.shape)

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


class testing_data(Dataset):
    def __init__(
        self,
        test_dir,
        classes=None,
        augmentation=None,
        preprocessing=None,
    ):

        self.ids = os.listdir(test_dir)
        self.filtered = [file for file in self.ids if file.endswith(".placeholder")]

        # Remove filtered
        for file in self.filtered:
            self.ids.remove(file)

        self.images_fps = [os.path.join(test_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return image
    def __len__(self):
        return len(self.ids)

#dataset = Dataset(train_img, train_mask, classes=['melanoma'])
#image, mask = dataset[3]
#visualize(image=image, mask=mask.squeeze())
