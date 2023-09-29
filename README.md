# Glacier Regression Detection Using Deep Learning

## Introduction
Glacier regression has been a concerning consequence of global climate change due to the increased negative impacts it has on the environment, fresh water resources and the rise in sea level. Accurately monitoring and understanding glacier regression is essential for both scientific research and informed climate policy decisions. Over the years, the application of Deep Learning methods in Remote Sensing and Earth Observation has brought about exciting possibilities for detecting and studying glacier regression with unprecedented precision. 
This project aims to use Deep Learning methods in detecting glacier regression. We will also explore how Neural Networks and Machine Learning algorithms can process vast volumes of satellite imagery to identify changes in glacier extents over time.

## Study Area
To proficiently develop a glacier detection and extraction algorithm, it needs to detect and handle all kinds of different coastline morphologies such as huge Ice Shelves, ridged Glacier Tongues, and Solid rocks. Eight different regions which was spread across Antarctica and Greenland for testing and training data were selected.

## Data 
The data used to train and test our Deep Learning Algorithm was Sentinel-2 MSI Level 1C product scenes from the study regions.

## Method
### Training and Testing Data Processing
For the deep learning algorithm to work proficiently, we required a Dataset, Binary Masks and Labels. Firstly a script was written by us to create binary masks of the scenes which we downloaded. Afterwards, we resized and masked the data collection. Then,  labels were generated from the masked scenes and were resized to be used in the algorithm. Here is an example of one the scenes and its binary mask.

<p align="center">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/94d853fd-7f73-47d2-8580-841dafadbd11.png" align="left" width="385" height="385">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/a4914c84-0a10-47cd-bfb9-d11b5ba41b3c.png" align="right" width="385" height="385">
</p>



### U-Net Architecture
To extract the border between the two classes, Land Ice and Ocean, a classifier was used by us which takes the pixel value as well as the spatial context into the account. Furthermore, basic structure from the originally developed U-Net was chosen from https://github.com/mmorphew/unet_remote_sensing & https://github.com/karolzak/keras-unet and modified the architecture for our purpose.

### Training
The model was trained by us on **3 different epoch values**, **2**, **10** and **20** to make a comparative analysis on which number of epochs yields the best model. Training the model on **2 epochs** generated some interesting results. The graph indicated a decrease in log loss within the loss and validation loss while having a minimal increase in the categorical accuracies. The best model was at the lowest point on the validation loss as shown in the graph below (left). The model trained on **10 epochs** had  a much higher categorical accuracies as compared to the model trained on **2 epochs**, with the categorical accuracy being slightly better than the validation accuracy. Furthermore, undulations in the validation loss was observed with the best model being close to the categorical accuracy as shown in the graph below (central).A huge spike in the validation loss on the model trained on **20 epochs** was seen. Nevertheless, the categorical accuracies had higher values than the models trained on **2 and 10 epochs**. The **20 epochs** model is on the (right) below.

<p align="center">
  <img src="https://github.com/Christobaltobbin/Classification-Land_cover_Analysis/assets/116877317/401a8583-5226-46fe-8725-08603b09b483.png" align="left" width="250" height="250">
  <img src="https://github.com/Christobaltobbin/Classification-Land_cover_Analysis/assets/116877317/5fca4542-0938-4aad-a541-5916d5e6962b.png" align="center" width="250" height="250">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/5743bf84-6054-41d9-aee2-26577d1998e9.png" align="right" width="250" height="250">
</p>

## Results
The predicted results of our three seperately trained models have very similar results. From the results we have 6 images, the Remote Sensing Image, Green Image, Near Infra-red Image (NIR), Image Mask, Ocean and Ice Binary Mask images. The model is not able to predict the Green and NIR images well as we only used scenes with one band from the data collection to train the model. The model is not able to predict the right image mask for the remote sensing image. We estimate that this is possibly due to low accuracies of the trained model, augmenting of the training data with fixed indices or training the model on a single band image instead of multiband images. Below are the predicted results of the trained models; 2 epochs model image (left), 10 epochs model image (central) and 20 epoch model image (right):

<p align="center">
  <img src="https://github.com/Christobaltobbin/Classification-Land_cover_Analysis/assets/116877317/017ff44e-9b4b-4154-82af-fa975e9eaf21.png" align="left" width="250" height="250">
  <img src="https://github.com/Christobaltobbin/Classification-Land_cover_Analysis/assets/116877317/efe7abfb-13a4-4418-a6e1-f72e938d9f74.png" align="center" width="250" height="250">
  <img src="https://github.com/Christobaltobbin/Drought_Assessment/assets/116877317/e474de31-8888-418d-8ac1-91ed511f7015.png" align="right" width="250" height="250">
</p>

*Choosing different optimizers and different loss functions could potentially influence the accuracies of the model.*

## Accuracy Assessment
The following were the accuracies for the training models. First, the model was trained on **2 epochs**, which showed a test loss of **73%** and a test accuracy of **57%**. Then, The model trained on **10 epochs** gave a test loss of **72.6%** and a test accuracy of **60.9%**. Subsequently, the model trained on **20 epochs** generated a test loss of **70.1%** and a test accuracy of **61.2%**. Comparing the three models, we conclude that the model trained on **20 epochs** has the highest accuracy. The precision score for all 3 models are **52.3%**, **51.3%** and **52.1%** respectively.

## Conclusion
 U-Net architecture can succesfully be used for detecting the changes in the glaciers over the years for various regions. Choosing different optimizers, loss functions and labelling techniques may influence the outcome of the model. Furthermore, assigning more augmentations with different methods may also leverage the outcome of the architecture. 

 The accuracy for change detection may vary with quality and the quantity of the training data. Introducing advanced techniques such as Transfer Learning, Recurrent Neural Networks (RNNs) and Optical Flow Methods, may further improve the performance of the model. 
