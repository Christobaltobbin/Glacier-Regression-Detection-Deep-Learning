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
For the deep learning algorithm to work proficiently we needed to have our dataset, binary masks and labels. Therefore we first wrote a script to create binary masks of the scenes we had downloaded. We then went ahead and also resized our scenes and masked scenes. We also generated labels from the masked scenes and resized them to be used in the algorithm. 

### U-Net Architecture
To extract the border between the two classes, land ice and ocean, we use a classifier which takes the pixel value as well as the spatial context into account. We chose the basic structure from the originally developed U-Net from Ronneberger et al. and modified the architecture for our purpose. Main modifications include (1) the usage of bigger input tiles; (2) starting with 32 feature channels (instead of 64) and increasing only to 512 (instead of 1024); (3) including drop out; and (4) feeding four input channels instead of one. Figure 2 visualizes our modified U-net architecture with four down-sampling units (red arrows) in the encoder block and four up-sampling units (green arrows) in the decoder block as well as skip-connections (black arrows). In the following, we explain the basic function of the U-Net, hyperparameter selection, and why we entered modifications.
