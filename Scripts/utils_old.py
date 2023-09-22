import cv2
import os
import tifffile # for reading tiff files. There are other modules that can do this, but tifffile is most stable on Windows
import numpy as np # for array handling
import matplotlib.pyplot as plt # for QC
import glob # to gather up image filepath lists
import rasterio
from rasterio.plot import show
import scipy # same
import imagecodecs

# creating a function for reading the training data
def read_data(files_path):
    top_list = glob.glob(files_path)
    top_list = np.sort(top_list)
    print(top_list)
    return top_list

# Function to Create arrays for resizing
def create_arrays(Image_resize_value:int, Number_of_channels:int):
    imgResize = Image_resize_value
    channels = Number_of_channels

    # Define the directory where you want to save the individual images
    save_dir_train = './train_images/'
    save_dir_test = './test_images/'

    # Create directories if they don't exist
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    return save_dir_train, save_dir_test

# Function to resize the images
def resize_images(resize_value, no_of_channels, data_list, output_file_path):

    for i in range(len(data_list)):
        img0 = tifffile.imread(data_list[i])  # Read the image
        img_reshaped = cv2.resize(img0, (resize_value, no_of_channels))  # Resize it

        # Local normalization & standardization of the image values
        img_norm = np.clip((img_reshaped - img_reshaped.mean()) / (0.5 * img_reshaped.std()), 0, 1)

        # Save the individual reshaped image as TIFF
        save_path = os.path.join(output_file_path, f'image_{i}.tif')
        tifffile.imwrite(save_path, img_norm)

def plot_label(label_list_path, index_number):
    img = tifffile.imread(label_list_path[index_number])
    plt.imshow(img, cmap='Blues')


