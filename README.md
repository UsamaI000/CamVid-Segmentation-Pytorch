# Semantic Segmentation Pytorch

## Semantic Segmentation
Basically, segmentation is a process that partitions an image into regions. It is an image processing approach that allows us to separate objects and textures in images. Segmentation is especially preferred in applications such as remote sensing or tumor detection in biomedicine.
The goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we’re predicting for every pixel in the image, this task is commonly referred to as dense prediction.
Note that unlike the classifcation tasks, the expected output in semantic segmentation are not just labels and bounding box parameters. The output itself is a high resolution image (typically of the same size as input image) in which each pixel is classified to a particular class. Thus it is a pixel level image classification.

## Guide to use
  - Download the files
  - Open Config.py and adjust it according to your setup and adjust related file paths in it.
  - To modify training loss and optimizer open train.py and edit it.
  - To start training run train.py
  - To test your model or get predictions of unseen data run test.py and predict.py

## Dataset
  <p> The data used for semantic Segmentation is CamVid Dataset. The dataset has 367 Training images, 101 Validation images and 232 Test images</p>
  <b> Link: https://www.kaggle.com/carlolepelaars/camvid </b> <br/>

  ### Training
  <p> We Trained the U-Net model on the dataset </p>
  <br/>
  <p align="center"> <img width=700 height= 350 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/unet.png"> </p>

## Training Setup
Following configurations were used for final model training.
  - Batch Size: 4
  - Learning rate: 0.0005
  - Optimizer: SGD
  - Loss: Focal Loss
  - Metric: IoU

## Training Results
The plot for Training of model is shown below.
   <p> The Loss Curve during Training </p>
   <p align="left"> <img src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/loss.png"> </p>
   <p> The Original Images were </p>
   <p align="left"> <img width=350 height= 120 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/train_orig.png"> </p>
   <p> The Original Masks were </p>
   <p align="left"> <img width=350 height= 120 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/train_mask.png"> </p>
   <p> The Predicted Masks were </p>
   <p align="left"> <img width=350 height= 120 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/train_pred.png"> </p>
   

## Test Results
The plot for Test predictions of model are shown below.
   <p> The Original Images were </p>
   <p align="left"> <img width=400 height= 150 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/test_orig.png"> </p>
   <p> The Original Masks were </p>
   <p align="left"> <img width=400 height= 150 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/test_mask.png"> </p>
   <p> The Predicted Masks were </p>
   <p align="left"> <img width=400 height= 150 src="https://github.com/UsamaI000/CamVid-Segmentation-Pytorch/blob/master/images/test_pred.png"> </p>
