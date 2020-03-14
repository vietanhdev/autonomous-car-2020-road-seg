import os
import cv2
import numpy as np
import pathlib
import shutil
import random


IMG_SIZE = (320, 240)
DATA_DIR = './data/'
RAW_DIR = './data/raw'

x_train_dir = os.path.join(DATA_DIR, 'train/images')
y_train_dir = os.path.join(DATA_DIR, 'train/masks')

x_valid_dir = os.path.join(DATA_DIR, 'val/images')
y_valid_dir = os.path.join(DATA_DIR, 'val/masks')

x_test_dir = os.path.join(DATA_DIR, 'test/images')
y_test_dir = os.path.join(DATA_DIR, 'test/masks')


masks = [file for file in os.listdir(os.path.join(RAW_DIR, 'masks')) if file.endswith('.png')]
random.seed(42)
random.shuffle(masks)

train_ends = int((len(masks)+1)*.92)
val_ends = int((len(masks)+1)*.98)

subset_masks = {
    "train": [],
    "test": [],
    "val": []
}
subset_masks["train"] = masks[:train_ends]
subset_masks["val"] = masks[train_ends:val_ends]
subset_masks["test"] = masks[val_ends:]


print(len(subset_masks["train"]))
print(len(subset_masks["val"]))
print(len(subset_masks["test"]))

for subset in ["train", "val", "test"]:
    subset_path = os.path.join(DATA_DIR, subset)
    if os.path.exists(subset_path) and os.path.isdir(subset_path):
        shutil.rmtree(subset_path)
    pathlib.Path(os.path.join(DATA_DIR, subset, "images")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(DATA_DIR, subset, "masks")).mkdir(parents=True, exist_ok=True)

    for mask in subset_masks[subset]:
        # shutil.copyfile(os.path.join(RAW_DIR, "images", mask), os.path.join(DATA_DIR, subset, "images", mask))
        # shutil.copyfile(os.path.join(RAW_DIR, "masks", mask), os.path.join(DATA_DIR, subset, "masks", mask))

        img = cv2.imread(os.path.join(RAW_DIR, "images", mask))
        img = cv2.resize(img, IMG_SIZE)
        cv2.imwrite( os.path.join(DATA_DIR, subset, "images", mask), img)

        mask_img = cv2.imread(os.path.join(RAW_DIR, "masks", mask))
        mask_img = cv2.resize(mask_img, IMG_SIZE)
        cv2.imwrite( os.path.join(DATA_DIR, subset, "masks", mask), mask_img)
