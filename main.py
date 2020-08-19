import os
import load_dataset

DATA_DIR = "Data/"
train_img = os.path.join(DATA_DIR,"Training_input")
train_mask = os.path.join(DATA_DIR,"Training_annot")

dataset = load_dataset.Dataset(train_img, train_mask, classes=['melanoma'])
image, mask = dataset[3]

load_dataset.visualize(image= image, mask = mask)

