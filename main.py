import torch
import numpy as np
import timeit
import math
from PIL import Image
import os
import matplotlib.pyplot as plt

train_path = "/Users/ardameric/Desktop/5.Y/CENG 483/ceng483_hw3/src/train/images"
val_path = "/Users/ardameric/Desktop/5.Y/CENG 483/ceng483_hw3/src/val/images"


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


if __name__ == '__main__':
    train_images = load_images_from_folder(train_path)
    plt.imshow(train_images[0])
    plt.show()



