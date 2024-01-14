"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/14 11:44
 @Author  : Ivan Mao
 @File    : transfer.py
 @Description : Transfer the model from GPU to CPU
"""
from itertools import product
import torch

for o_model, type in product(['CNN', 'DigitRecognizerNN'], ['origin', 'argument', 'normal']):
    model = torch.load(f"./checkpoint/{o_model}_cuda_{type}.pth")
    model.to('cpu')
    torch.save(model, f"./checkpoint/{o_model}_cpu_{type}.pth")