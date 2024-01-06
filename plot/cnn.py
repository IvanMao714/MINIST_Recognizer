"""
!/usr/bin/env python
 -*- coding: utf-8 -*-
 @CreateTime    : 2024/1/4 16:22
 @Author  : Ivan Mao
 @File    : cnn.py
 @Description : 
"""
import sys

from lib.plotneuralnet.pycore.tikzeng import *

sys.path.append('../lib/plotneuralnet/')

# defined your arch
arch = [
    to_head( '../lib/plotneuralnet/' ),
    to_cor(),
    to_begin(),
    to_Conv("conv1", 28, 16, offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=4),
    to_Relu("conv1_relu", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=1),
    to_Pool("conv1_pool", offset="(0,0,0)", to="(conv1_relu-east)", height=28, depth=28, width=1),

    to_connection("(0,0,0)", "(3,0,0)"),

    to_Conv("conv2", 28, 32, offset="(3,0,0)", to="(0,0,0)", height=14, depth=14, width=6),
    to_Relu("conv2_relu", offset="(0,0,0)", to="(conv2-east)", height=14, depth=14, width=1),
    to_Pool("conv2_pool", offset="(0,0,0)", to="(conv2_relu-east)", height=14, depth=14, width=1),

    to_connection("(3,0,0)", "(6,0,0)"),

    to_SoftMax("soft1", 10, "(0,0,0)", "(conv2_pool-east)", caption="SOFT"),

    # to_Conv("conv2", 1, 16, offset="(3,0,0)", to="(0,0,0)", height=28, depth=28, width=8),
    # to_ConvConvRelu("conv2_relu", s_filer=16, n_filer=(16, 16), offset="(0,0,0)", to="(3,0,0)", height=28, depth=28,
    #                 width=(2, 2), caption="Conv1"),
    # to_Pool("conv2_pool", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=1),

    # to_Conv("conv2", 16, 1, offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=2 ),
    # to_ConvConvRelu("conv2_relu", s_filer=16, n_filer=(16, 16), offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=(2, 2), caption="Conv1"),
    # to_Pool("conv2_pool", offset="(0,0,0)", to="(conv1-east)", height=28, depth=28, width=1),

    # to_Conv("conv2", 128, 64, offset="(1,0,0)", to="(pool1-east)", height=32, depth=32, width=2 ),
    # to_connection( "pool1", "conv2"),
    # to_Pool("pool2", offset="(0,0,0)", to="(conv2-east)", height=28, depth=28, width=1),
    # to_SoftMax("soft1", 10 ,"(3,0,0)", "(pool1-east)", caption="SOFT"  ),
    # to_connection("pool2", "soft1"),
    # to_Sum("sum1", offset="(1.5,0,0)", to="(soft1-east)", radius=2.5, opacity=0.6),
    # to_connection("soft1", "sum1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()