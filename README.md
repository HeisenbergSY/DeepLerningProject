# Pneumonia Detection using Deep Learning

This project aims to develop a deep learning model for the classification of pneumonia from chest X-ray images. The project involves data augmentation, handling class imbalance, model training, evaluation, and inference.

## Project Overview

The goal of this project is to classify chest X-ray images into two categories: Normal and Pneumonia. We will use Convolutional Neural Networks (CNNs) and various data augmentation techniques to improve model performance.

## Dataset

The dataset used for this project consists of chest X-ray images categorized into two classes: Normal and Pneumonia. The dataset can be downloaded from:(https://drive.google.com/drive/folders/1N9D68Uj6Y3R8_iYAE_dnP9J5BXUiDXRy?usp=sharing).

### Data Structure

The dataset should be organized as follows:
data/
train/
NORMAL/
PNEUMONIA/
test/
NORMAL/
PNEUMONIA/

## Requirements

The following libraries are required to run this project:

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- pillow

## Changes Made
- Added data augmentation (rotation, flipping, and zooming) to improve model generalization.
- Updated training script to include data augmentation.
- To handle class imbalance undersampling of minority class has been implemented
- More preprocessing step: Histogram Equalization
- Doing test with different lerning rate. Best lerning rate 0.001
- Implement early stopping to not overfit the model

## How to run
- 3 Models has been trained. For each model 3 files has been uploaded. Model-run.py, model.txt, and model-Model.pth. txt file contains training results of that model, -run is code ready to be run to test the training and -Model.pth is the final model with the best results
## Results
- **Each run is logged in training_log file. The log files include information about the used parameters and the results. Log files are found in TEMPLATE_PROJECT folder.**


