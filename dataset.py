"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/4 10:49
 @Author  : Ivan Mao
 @File    : dataset.py
 @Description : 
"""
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class MNIST_data(Dataset):
    """MNIST data set"""

    def __init__(self, X, y=None,
                 transform=transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,))])
                 , trans = False):

        # if len(df.columns) == n_pixels:
        #     # test data
        #     self.X = df.values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
        #     self.y = None
        # else:
        #     # training data
        #     self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.uint8)[:, :, :, None]
        #     self.y = torch.from_numpy(df.iloc[:, 0].values)
        self.X = X
        self.y = y
        if trans:
            self.transform = transform
        else:
            self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.transform(self.X[idx]), self.y[idx]
        else:
            return self.transform(self.X[idx])