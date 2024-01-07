"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/7 12:08
 @Author  : Ivan Mao
 @File    : drnn.py
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
    to_BatchNorm("bn1", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=4, caption="BN1"),
    # to_Relu("conv1_relu", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=4),
    # to_Pool("conv1_pool", offset="(0,0,0)", to="(conv1_relu-east)", height=14, depth=14, width=4),

    to_connection("(0,0,0)", "(3,0,0)"),

    to_Conv("conv2", 28, 64, offset="(3,0,0)", to="(0,0,0)", height=28, depth=28, width=8, caption="Conv2"),
    to_BatchNorm("bn2", offset="(3.8,0,0)", to="(conv1-east)", height=28, depth=28, width=8, caption="BN2"),

    to_connection("(4,0,0)", "(8,0,0)"),

    to_Conv("conv2", 28, 128, offset="(8,0,0)", to="(0,0,0)", height=28, depth=28, width=10, caption="Conv3"),
    to_BatchNorm("bn2", offset="(9.2,0,0)", to="(conv1-east)", height=28, depth=28, width=10, caption="BN3"),

    to_connection("(10,0,0)", "(14,0,0)"),

    to_Pool("conv2_pool", offset="(14,0,0)", to="(0,0,0)", height=14, depth=14, width=10, caption="MaxPool1"),

    to_connection("(14,0,0)", "(17,0,0)"),

    to_Relu("dropout1", offset="(17,0,0)", to="(0,0,0)", height=14, depth=14, width=10, caption="Dropout1"),
    to_Relu("dropout2", offset="(19,0,0)", to="(0,0,0)", height=14, depth=14, width=10, caption="Dropout2"),

    to_connection("(19,0,0)", "(24,0,0)"),

    to_Fc("fc1", 64, offset="(24,0,0)", to="(0,0,0)", height=1, depth=20, width=1, caption="FC1"),

    to_connection("(24,0,0)", "(27,0,0)"),

    to_Fc("fc2", 10, offset="(27,0,0)", to="(0,0,0)", height=1, depth=10, width=1, caption="FC2"),

    to_connection("(27,0,0)", "(30,0,0)"),

    to_SoftMax("soft1", 10,"(30,0,0)", "(0,0,0)", caption="SOFT"),

    # to_Fc("fc1", 10, offset="(12,0,0)", to="(0,0,0)", height=1, depth=10, width=1),

    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()