"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/3 22:00
 @Author  : Ivan Mao
 @File    : pipeline.py
 @Description : 
"""
import os

import pandas as pd
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.transforms import transforms
import torch.nn.functional as F

from tqdm import tqdm

import transform
from dataset import MNIST_data
from download import load
from model.CNN import CNN
from model.DigitRecognizerNN import DigitRecognizerNN


def train(dataloader, model, loss_fn, optimizer, device):
    # Get the total size of the dataset
    size = len(dataloader.dataset)
    # Set the model in training mode
    model.train()
    # Initialize variables to track loss and accuracy
    running_loss = 0.0
    correct = 0
    total = 0

    # Create a progress bar using tqdm to visualize training progress
    with tqdm(total=len(dataloader), desc="Training", unit="batch") as pbar:
        # Iterate through each batch in the dataloader
        for batch, (X, y) in enumerate(dataloader):
            # Move data to the specified device (GPU or CPU)
            X, y = X.to(device), y.to(device)

            # Zero the gradients to prepare for backpropagation
            optimizer.zero_grad()
            # Forward pass: compute predictions
            pred = model(X)
            # Calculate the loss
            loss = loss_fn(pred, y)
            # Backpropagation: compute gradients
            loss.backward()
            # Update model parameters using the optimizer
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            # Calculate predicted labels by finding the index of the maximum value along dim=1
            predicted = torch.argmax(pred, dim=1)
            # Update total number of samples processed
            total += y.size(0)
            # Update number of correctly predicted samples
            correct += (predicted == y).sum().item()

            # Update the progress bar
            pbar.update(1)
            # Update displayed metrics in the progress bar
            pbar.set_postfix({'loss': running_loss / (batch + 1), 'accuracy': 100 * correct / total})

    # Calculate epoch-level loss and accuracy
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / size

    # Print epoch-level metrics
    print(f"End of Epoch: Avg loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}%\n")
    return epoch_loss, epoch_acc


def process_data(type='train', argument=False,trans=False):
    # Convert the NumPy arrays to PyTorch tensors
    train_images, train_labels, test_images, test_labels = load()
    train_images_tensor = torch.tensor(train_images).reshape(-1, 1, 28, 28)
    train_labels_tensor = torch.tensor(train_labels)
    test_images_tensor = torch.tensor(test_images).reshape(-1, 1, 28, 28)
    test_labels_tensor = torch.tensor(test_labels)

    # Normalize the images
    # train_images_tensor = train_images_tensor.float().reshape(-1, 1, 28, 28) / 255.0
    # test_images_tensor = test_images_tensor.float().reshape(-1, 1, 28, 28) / 255.0


    # Create DataLoader for the training data
    if type == 'train':
        images_tensor = train_images_tensor
        labels_tensor = train_labels_tensor
    elif type == 'test':
        images_tensor = test_images_tensor
        labels_tensor = test_labels_tensor
    else:
        raise ValueError('type should be train or test')
    argument_dataset = MNIST_data(images_tensor, labels_tensor, transform= transforms.Compose(
                            [transforms.ToPILImage(), transform.RandomRotation(degrees=20), transform.RandomShift(3),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
                                  ,trans= trans)
    dataset = MNIST_data(images_tensor, labels_tensor)
    if argument:
        dataset = ConcatDataset([dataset, argument_dataset])
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    return dataloader, labels_tensor


def trainer(dataloader, model, device, mark=None):
    if model == 'cnn':
        model = CNN().to(device)
    else:
        model = DigitRecognizerNN().to(device)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.8, weight_decay=0.000025)
    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    epochs = 10
    epoch_loss_list, epoch_acc_list = [],[]

    # Loop through each epoch
    for epoch in range(epochs):
        # Print the current epoch number
        print(f"Epoch {epoch + 1}")

        # Execute the training process for the current epoch
        epoch_loss, epoch_acc = train(dataloader, model, loss_fn, optimizer, device)

        # Update the learning rate using the scheduler after completing an epoch
        scheduler.step()

        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
    data = {f'loss': epoch_loss_list, f'acc': epoch_acc_list}
    df = pd.DataFrame(data)
    df.to_csv(f'./checkpoint/{model.__class__.__name__}_{device}_{mark}.csv', index=False)
    # Save the trained model's state dictionary to a file named "model.pth"
    if not os.path.exists("./checkpoint"):
        os.makedirs("./checkpoint")
    torch.save(model, f"./checkpoint/{model.__class__.__name__}_{device}_{mark}.pth")
    print(f"Saved PyTorch Model State to {model.__class__.__name__}.pth")


def evaluate(data_loader,model):
    model.eval()
    loss = 0
    correct = 0

    for data, target in data_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.to('cuda')
            target = target.to('cuda')

        output = model(data)

        loss += F.cross_entropy(output, target, size_average=False).data[0]

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(data_loader.dataset)

    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))