import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

def get_image(image_path, width, height, mode):
    """
        Read image from image_path
        :param image_path: Path of image
        :param width: Width of image
        :param height: Height of image
        :param mode: Mode of image
        :return: Image data
        """
    image = Image.open(image_path)
    return np.array(image.convert(mode))        

def get_batch(image_files, width, height, mode):
    data_batch = np.array([get_image(sample_file, width, height, mode) 
                           for sample_file in image_files]).astype(np.float32)

    # Make sure the images are in 4 dimensions
    if len(data_batch.shape) < 4:
        data_batch = data_batch.reshape(data_batch.shape + (1,))

    return data_batch

class Dataset(object):
    def __init__(self, dataset_name, data_files, data_size):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_NF_NAME = 'nf'
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = data_size
        IMAGE_HEIGHT = data_size

        if dataset_name == DATASET_NF_NAME:
            self.image_mode = 'RGB'
            image_channels = 3
            
        elif dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels
    
    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size
            
            yield data_batch

#             yield data_batch / IMAGE_MAX_VALUE - 0.5 # normalize
