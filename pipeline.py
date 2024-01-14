"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/3 22:00
 @Author  : Ivan Mao
 @File    : pipeline.py
 @Description : 
"""
import os
from itertools import product

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from torchvision.transforms import transforms
from tqdm import tqdm

import transform
from dataset import MNIST_data
from download import load, load_mnist_data
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


def process_data(type='train', argument=False, trans=False):
    # Convert the NumPy arrays to PyTorch tensors
    if type == 'train':
        images, labels = load_mnist_data('./data/MINIST/train')
    elif type == 'test':
        images, labels = load_mnist_data('./data/MINIST/test')
    images_tensor = torch.tensor(images).reshape(-1, 1, 28, 28)
    labels_tensor = torch.tensor(labels).long()

    # Create DataLoader for the training data
    # else:
    #     raise ValueError('type should be train or test')
    argument_dataset = MNIST_data(images_tensor, labels_tensor, transform=transforms.Compose(
        [transforms.ToPILImage(), transform.RandomRotation(degrees=20), transform.RandomShift(3),
         transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
                                  , trans=trans)
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
    epoch_loss_list, epoch_acc_list = [], []

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


def validate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(total=len(dataloader), desc="Validation", unit="batch") as pbar:
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)

                val_loss += loss.item()
                predicted = torch.argmax(pred, dim=1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

                pbar.update(1)
                pbar.set_postfix({'val_loss': val_loss / (batch + 1), 'val_accuracy': 100 * correct / total})

    epoch_loss = val_loss / len(dataloader)
    epoch_acc = 100 * correct / size
    return epoch_loss, epoch_acc


def validater(device, dataloader):
    loss_fn = nn.CrossEntropyLoss()
    data = pd.DataFrame(columns=['loss', 'acc', 'type', 'model'])
    for model, type in product(['CNN', 'DigitRecognizerNN'], ['origin', 'argument', 'normal']):
        model = torch.load(f'./checkpoint/{model}_{device}_{type}.pth')
        loss, acc = validate(dataloader, model, loss_fn, device)
        print(model.__class__.__name__, type)
        print(f"End of Validation: Avg loss: {loss:.4f}, Accuracy: {acc:.4f}%\n")
        data.loc[len(data)] = [loss, acc, type, model.__class__.__name__]
        # data = data.append(pd.DataFrame({'loss': loss, 'acc': acc, 'type': type, 'model': model.__class__.__name__}))
    data.to_csv(f'./checkpoint/validate_{device}.csv', index=False)

