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

