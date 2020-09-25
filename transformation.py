import albumentations as albu
import cv2

def get_training_augmentation():
    train_transform = [
        albu.Resize(224,224,interpolation=cv2.INTER_AREA),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():

    test_transform= [

        albu.Resize(224,224,interpolation=cv2.INTER_AREA),

    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)