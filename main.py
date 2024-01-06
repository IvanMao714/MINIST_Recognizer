"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/3 22:04
 @Author  : Ivan Mao
 @File    : main.py
 @Description : 
"""
import torch
from itertools import product

from model.DigitRecognizerNN import DigitRecognizerNN
from pipeline import process_data, trainer, evaluate

# Instantiate the model and move it to the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose device based on availability
# train_dataloader, train_labels_tensor = process_data(type='train', argument=False, trans=False)
# trainer(train_dataloader, 'drnn', device, mark='origin')

# for model, mark in product(['cnn', 'drnn'], ['origin', 'argument','normal']):
#     print(model, mark)
#     if mark== 'origin':
#         train_dataloader, train_labels_tensor = process_data(type='train', argument=False, trans=False)
#     elif mark=='argument':
#         train_dataloader, train_labels_tensor = process_data(type='train', argument=True, trans=True)
#     else:
#         train_dataloader, train_labels_tensor = process_data(type='train', argument=False, trans=True)
#     trainer(train_dataloader, model, device, mark=mark)

model = DigitRecognizerNN()
model.to(device)
model.load_state_dict(torch.load('./checkpoint/DigitRecognizerNN_cuda.pth'))
print(model)
test_dataloader, test_labels_tensor = process_data(type='test', argument=False, trans=True)
# trainer(train_dataloader, model, device, mark=mark)
evaluate(test_dataloader, model)
# test_dataloader, test_labels_tensor = process_data(type='test', argument=False, trans=False)
# model = torch.load('./checkpoint/DigitRecognizerNN_cuda.pth')
# print(model)
# evaluate(test_dataloader,model)