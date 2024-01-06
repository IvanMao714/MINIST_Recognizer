"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/4 16:03
 @Author  : Ivan Mao
 @File    : augmentation.py
 @Description : 
"""
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=10, # Rotating randomly the images up to 25°
                             width_shift_range=0.05, # Moving the images from left to right
                             height_shift_range=0.05, # Then from top to bottom
                             shear_range=0.10,
                             zoom_range=0.05, # Zooming randomly up to 20%
                             zca_whitening=False,
                             horizontal_flip=False,
                             vertical_flip=False,
                            fill_mode = 'nearest')