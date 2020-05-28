from os.path import splitext
from os import listdir
import numpy as np
import pandas as pd
from pandas import DataFrame
import os 
import skimage 
from skimage import io 
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """        
        data_frame = pd.read_csv(csv_file)

        ## Local images only  
        
        local_files_df = DataFrame([os.path.splitext(filename)[0] for filename in os.listdir('/Users/abharani/Documents/myworkspace/cs231n_project/data/train_images/') if filename.endswith(".tiff")], columns=["image_id"])
        self.data_frame = local_files_df.join(data_frame.set_index('image_id'), on='image_id')
        ##
        self.root_dir = root_dir
        self.transform = transform
        logging.info('Creating dataset with {} examples'.format(len(self.data_frame)))
        
    def __len__(self):
        return len(self.data_frame)

    @classmethod
    def preprocess(cls, image):
        WINDOW_SIZE = 128
        STRIDE = 64
        K = 16
        
        image, best_coordinates, best_regions = generate_patches(image, window_size=WINDOW_SIZE, stride=STRIDE, k=K)
        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)
        
        return glued_image


    def __getitem__(self, idx):
        
        image_id = self.data_frame.iloc[idx]['image_id']
        
        image_file_path = os.path.join(self.root_dir,image_id + ".tiff")
        
        image = skimage.io.MultiImage(image_file_path)[-1]
        image = np.array(image)

        isup_grade = self.data_frame.iloc[idx]['isup_grade']

        image = self.preprocess(image)

        return {'image': torch.from_numpy(image), 'isup_grade' : isup_grade}


def compute_statistics(image):
    """
    Args:
        image                  numpy.array   multi-dimensional array of the form WxHxC
    
    Returns:
        ratio_white_pixels     float         ratio of white pixels over total pixels in the image 
    """
    width, height = image.shape[0], image.shape[1]
    num_pixels = width * height
    
    num_white_pixels = 0
    
    summed_matrix = np.sum(image, axis=-1)
    # Note: A 3-channel white pixel has RGB (255, 255, 255)
    num_white_pixels = np.count_nonzero(summed_matrix > 620)
    ratio_white_pixels = num_white_pixels / num_pixels
    
    green_concentration = np.mean(image[1])
    blue_concentration = np.mean(image[2])
    
    return ratio_white_pixels, green_concentration, blue_concentration

def select_k_best_regions(regions, k=20):
    """
    Args:
        regions               list           list of 2-component tuples first component the region, 
                                             second component the ratio of white pixels
                                             
        k                     int            number of regions to select
    """
    regions = [x for x in regions if x[3] > 180 and x[4] > 180]
    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]
    return k_best_regions

def generate_patches(image, window_size=200, stride=128, k=20):
    
#     image = skimage.io.MultiImage(slide_path)[-2]
#     image = np.array(image)
    
    max_width, max_height = image.shape[0], image.shape[1]
    regions_container = []
    i = 0
    
    while window_size + stride*i <= max_height:
        j = 0
        
        while window_size + stride*j <= max_width:            
            x_top_left_pixel = j * stride
            y_top_left_pixel = i * stride
            
            patch = image[
                x_top_left_pixel : x_top_left_pixel + window_size,
                y_top_left_pixel : y_top_left_pixel + window_size,
                :
            ]
            
            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)
            
            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)
            regions_container.append(region_tuple)
            
            j += 1
        
        i += 1
    
    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)
    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)
    
    return image, k_best_region_coordinates, k_best_regions

def display_images(regions, title):
    fig, ax = plt.subplots(5, 4, figsize=(15, 15))
    
    for i, region in regions.items():
        ax[i//4, i%4].imshow(region)
    
    fig.suptitle(title)
    
    
def get_k_best_regions(coordinates, image, window_size=512):
    regions = {}
    for i, tup in enumerate(coordinates):
        x, y = tup[0], tup[1]
        regions[i] = image[x : x+window_size, y : y+window_size, :]
    
    return regions


def glue_to_one_picture(image_patches, window_size=200, k=16):
    side = int(np.sqrt(k))
    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)
        
    for i, patch in image_patches.items():
        x = i // side
        y = i % side
        image[
            x * window_size : (x+1) * window_size,
            y * window_size : (y+1) * window_size,
            :
        ] = patch
    
    return image