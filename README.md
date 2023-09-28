# Glacier Regression Detection Using Deep Learning

## Introduction
Glacier regression has been a concerning consequence of global climate change due to the increased negative impacts it has had on the environment, fresh water resources and the rise in sea level. Accurately monitoring and understanding glacier regression is essential for both scientific research and informed climate policy decisions. Over the years, the application of deep learning methods in remote sensing and Earth observation has brought about exciting possibilities for detecting and studying glacier regression with unprecedented precision. 
This project aims to use deep learning methods in detecting glacier regression. We will also explore how neural networks and machine learning algorithms can process vast volumes of satellite imagery to identify changes in glacier extents over time.

## Study Area
To proficiently develop a glacier detection and extraction algorithm, it needs to detect and handle all kinds of different coastline morphologies such as huge ice shelves, ridged glacier tongues, and solid rock (). In all we selected eight different regions which was spread across Antartica and Greenland for our testing and training data.

## Data 
The data used to train and test our deep learning algorithm was Sentinel 2 MSI Level 1C products scenes from the study regions.

## Method
### Training and Testing Data Processing
For the deep learning algorithm to work proficiently we needed to have our dataset, binary masks and labels. Therefore we first wrote a script to create binary masks of the scenes we had downloaded. We then went ahead and also resized our scenes and masked scenes. We also generated labels from the masked scenes and resized them to be used in the algorithm. Let us view one of the scenes and its binary mask.

<p align="center">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/94d853fd-7f73-47d2-8580-841dafadbd11.png" align="left" width="385" height="385">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/a4914c84-0a10-47cd-bfb9-d11b5ba41b3c.png" align="right" width="385" height="385">
</p>

### U-Net Architecture
To extract the border between the two classes, land ice and ocean, we use a classifier which takes the pixel value as well as the spatial context into account. We chose the basic structure from the originally developed U-Net from https://github.com/mmorphew/unet_remote_sensing & https://github.com/karolzak/keras-unet and modified the architecture for our purpose. Main modifications include :

### Training

## Results

## Accuracy Assessment
