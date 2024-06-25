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

## Results
- **Original Test Accuracy:** ~ 70.83%
- **Test Accuracy with Augmentation:** ~ 84.46%
- **Test Accuracy with Augmentation (Second run):** ~ 90.38%
- **Test Accuracy with Augmentation (Third run):** ~ 93.43%
