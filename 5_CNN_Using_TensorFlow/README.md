# Homework 5: CNN using TensorFlow

## Description

In this homework you will practice how to create Convolutional Neural Network in Python with TensorFlow. You need to understand how CNN works, including back propagation and gradient decent is performed in order to complete this homework successfully. The goals of this homework are:

o To understand the steps to train/test the classifier for image classification.
o Understand architecture of CNN and how to connect each layer together by using TensorFlow.
o To implement and understand CNN using TensorFlow.

## Instruction

In this homework, you need to fill the block of code in 1 python file, cnnTF.py.

- cnnTF.py: This is the main file that you will execute. It will read and processing
CIFAR10 dataset, and run models.

- There are 3 models:

o Simple Model, an example of a convolutional neural network which already
implemented.

o Complex Model, you have to implement model follow the architecture
below.

- 7x7 Conv with stride 2
- Relu activation
- 2x2 Max Pooling
- Fully connected with 1024 hidden neurons
- Relu activation
- Fully connected that map to 10 output classes
- You may use batch normalization technique.
- Multiple convolutional layers with different size of filter and stride.