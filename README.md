# Fruit Classification with PyTorch

This project implements a fruit classification model using PyTorch. The model is built on the pre-trained ResNet-18 architecture and fine-tuned to classify images from the Food-360 dataset. 

## Table of Contents

- [Overview](#overview)
- [Code Structure](#code-structure)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Inference](#inference)
- [Results](#results)

## Overview

The goal of this project is to classify images of fruits using a Convolutional Neural Network (CNN). It leverages the power of transfer learning by using a pre-trained ResNet-18 model, which helps improve classification accuracy with fewer training epochs.

## Code Structure

The project consists of the following files:

- `train.py`: Contains the code to load the dataset, train the model, and save the best performing model.
- `classify.py`: Loads the trained model and classifies a given image.
- `class_names.json`: Stores the class names corresponding to the dataset.
- `best_fruits_model.pth`: The saved model weights after training (generated during the training process).

### Key Components in `train.py`

1. **Data Transformations**:
   - Images are resized and normalized to prepare them for the model.
   - Data augmentation techniques can be added for improved generalization.

2. **Dataset Loading**:
   - The `ImageFolder` class is used to load images from the training and validation directories.

3. **Model Initialization**:
   - A pre-trained ResNet-18 model is loaded, and the final fully connected layer is modified to match the number of fruit classes.
   - A dropout layer is added to reduce overfitting.

4. **Training Loop**:
   - The model is trained for a specified number of epochs, with loss and accuracy calculated for both training and validation datasets.
   - The model is saved if validation accuracy improves.

### Key Components in `classify.py`

1. **Image Classification**:
   - The script loads a pre-trained model and applies the same transformations used during training to the input image.
   - It then predicts the class of the image and displays the result.

## Requirements

Make sure you have the following libraries installed:

- Python 3.6+
- PyTorch
- torchvision
- matplotlib

You can install the required libraries using pip:

pip install torch torchvision matplotlib

## Training the Model
To train the model, run:

python train.py

This will:

Load the dataset.

Train the model for a specified number of epochs.

Save the best model weights as best_fruits_model.pth.

## Inference
To classify a new fruit image, update the image_path variable in classify.py with the path to your image and run:

python classify.py

This will display the image along with the predicted class.

## Results
The model is capable of accurately classifying fruit images based on the training dataset. The results can be evaluated using validation accuracy metrics, which are printed during the training process.

