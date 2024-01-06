"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/3 21:36
 @Author  : Ivan Mao
 @File    : DigitRecognizerNN.py
 @Description : 
"""
from torch import nn
import torch.nn.functional as F


class DigitRecognizerNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2, padding=1)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.75)
        self.dropout2 = nn.Dropout(0.625)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 5 * 5, 64)  # Input size after max pooling
        self.fc2 = nn.Linear(64, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x))))  # Convolution, BatchNorm, Leakyleaky_relu, Pooling
        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Flatten the tensor for fully connected layers

        # Apply dropout
        x = self.dropout1(x)

        # Fully connected layers with Leakyleaky_relu activations
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout2(x)

        # Output layer without activation (to be handled by loss function)
        x = F.softmax(self.fc2(x))
        return x
