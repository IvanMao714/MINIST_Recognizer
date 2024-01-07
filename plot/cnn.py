"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/4 16:22
 @Author  : Ivan Mao
 @File    : cnn.py
 @Description : 
"""
import sys

from lib.plotneuralnet.pycore.blocks import block_Res
from lib.plotneuralnet.pycore.tikzeng import *

sys.path.append('../lib/plotneuralnet/')

# defined your arch
arch = [
    to_head( '../lib/plotneuralnet/' ),
    to_cor(),
    to_begin(),
    to_prospect_image( "../img/mnist_single.png", width=6, height=6),

    to_Conv("conv1", 28, 16, offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=4, caption="Conv1"),
    to_Relu("conv1_relu", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=4),
    to_Pool("conv1_pool", offset="(0,0,0)", to="(conv1_relu-east)", height=14, depth=14, width=4,  caption="MaxPool1"),

    to_connection("(0,0,0)", "(6,0,0)"),

    to_Conv("conv2", 28, 32, offset="(6,0,0)", to="(0,0,0)", height=14, depth=14, width=6, caption="Conv2"),
    to_Relu("conv2_relu", offset="(0,0,0)", to="(conv2-east)", height=14, depth=14, width=6, caption="ReLU2"),
    to_Pool( "conv2_pool", offset="(0,0,0)", to="(conv2_relu-east)", height=7, depth=7, width=6, caption="MaxPool2"),

    to_connection("(3,0,0)", "(12,0,0)"),

    to_Fc("fc1", 10, offset="(12,0,0)", to="(0,0,0)", height=1, depth=10, width=1, caption="FC1"),

    to_connection("(12,0,0)", "(15,0,0)"),

    to_SoftMax("soft1", 10,"(15,0,0)", "(0,0,0)", caption="SOFT"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()