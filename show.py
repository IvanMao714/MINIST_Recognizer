"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/4 11:18
 @Author  : Ivan Mao
 @File    : show.py
 @Description : Figure show
"""
import os
from itertools import product

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import seaborn as sns

from download import load
from transform import RandomRotation, RandomShift

train_images, train_labels, test_images, test_labels = load()
random_sel = np.random.randint(200, size=8)
# blue_palette = ["#ff287bfc", "#00b2ff", "#00dbf1","#6efacc"]
# sns.set_palette(sns.light_palette("#79C", reverse=True))
sns.set_palette(sns.color_palette("ch:start=.2,rot=-.3",4))
def show_image():
    """Show dataset images"""
    grid = make_grid(
        torch.Tensor(train_images[random_sel].reshape((-1, 28, 28))).unsqueeze(1), nrow=8)
    plt.rcParams['figure.figsize'] = (16, 2)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.savefig('./img/mnist_show.png')
    plt.show()


def static_count(train_labels=train_labels):
    """Show the statics of dataset"""
    train_labels = pandas.Series(train_labels)
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.bar(train_labels.value_counts().index, train_labels.value_counts())
    plt.xticks(np.arange(10))
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.grid('on', axis='y')
    plt.savefig('./img/mnist_count.png')
    plt.show()


def compose_img_show():
    """Show the composed images"""
    rotate = RandomRotation(20)
    shift = RandomShift(3)
    composed = transforms.Compose([RandomRotation(20),
                                   RandomShift(3)])
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = transforms.ToPILImage()(train_images[0].reshape((28, 28)).astype(np.uint8))
    for i, tsfrm in enumerate([rotate, shift, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        ax.imshow(np.reshape(np.array(list(transformed_sample.getdata())), (-1, 28)), cmap='gray')
    plt.savefig('./img/mnist_compose.png')
    plt.show()


def compose_img_show():
    """Show result of the composed images"""
    cnn = pd.read_csv('./checkpoint/CNN_cuda.csv')
    drnn = pd.read_csv('./checkpoint/DigitRecognizerNN_cuda.csv')
    acc_combined_data = pd.concat([cnn['acc'], drnn['acc']], ignore_index=False, axis=1)
    acc_combined_data.columns = ['cnn_acc', 'drnn_acc']
    loss_combined_data = pd.concat([cnn['loss'], drnn['loss']], ignore_index=False, axis=1)
    loss_combined_data.columns = ['cnn_loss', 'drnn_loss']

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=acc_combined_data.index, y='cnn_acc', data=acc_combined_data, marker='o', label='CNN')
    sns.lineplot(x=acc_combined_data.index, y='drnn_acc', data=acc_combined_data, marker='s', label='DRNN')

    plt.xlabel('Epoch', fontsize=16)
    plt.title('Comparison of Models: Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(title='Model', loc='lower right')
    plt.savefig('./img/mnist_acc_comparison.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=loss_combined_data.index, y='cnn_loss', data=loss_combined_data, marker='o', label='CNN')
    sns.lineplot(x=loss_combined_data.index, y='drnn_loss', data=loss_combined_data, marker='s', label='DRNN')

    plt.xlabel('Epoch', fontsize=16)
    plt.title('Comparison of Models: Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Model', loc='lower right')
    plt.savefig('./img/mnist_loss_comparison.png')
    plt.show()


def single_img_show():
    """Show single image"""
    plt.imshow(train_images[0].reshape((28, 28)), cmap='gray')
    plt.savefig('./img/mnist_single.png')
    plt.show()


def test_comparison():
    """Show test result"""
    data = pd.read_csv('checkpoint/validate_cuda_1.csv')
    plt.figure(figsize=(10, 5))
    plt.ylim(96, 100)
    sns.barplot(data=data, x='model', y='acc', hue='type')
    plt.title('Testing Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Type')
    plt.savefig('./img/mnist_test_comparison.png')
    plt.show()

def train_comparison():
    """Show train result"""
    data = pd.DataFrame(columns=['acc', 'loss', 'type', 'model'])
    for model, type in product(['CNN', 'DigitRecognizerNN'], ['origin', 'augment', 'normal']):
        train_data = pd.read_csv(f'checkpoint/{model}_cuda_{type}.csv')
        train_data['type'] = type
        train_data['model'] = model
        data = pd.concat([data, train_data], ignore_index=False, axis=0)
    #     print(train_data)
    # print(data)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x=data.index, y='acc', hue='type', style='model', markers=True)
    plt.ylim(50, 100)
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Type')
    plt.savefig('./img/mnist_train_comparison.png')
    plt.show()

    sns.barplot(data=data, x='model', y='acc', hue='type')
    plt.ylim(70, 100)
    plt.title('Traing Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Type')
    plt.savefig('./img/mnist_train_comparison_bar.png')
    plt.show()

    sns.barplot(data=data, x='model', y='loss', hue='type')
    # plt.ylim(70, 100)
    plt.title('Traing Loss Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.legend(title='Type')
    plt.savefig('./img/mnist_train_loss_comparison_bar.png')
    plt.show()




if __name__ == '__main__':
    # show_image()
    # static_count()
    # compose_img_show()
    # compose_img_show()
    # single_img_show()
    # test_comparison()
    train_comparison()
