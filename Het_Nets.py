#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:34:22 2022

@author: alain
"""

import torch
from torch import nn
import torch.nn.functional as F
import itertools
from FLAlgorithms.trainmodel.models1 import Net2

def get_reg_model(args):
    net_glob = MLPReg_tensor_out(dim_in=args.dim_latent).to(args.device)
    # specify the representation parameters (in w_glob_keys) and head parameters (all others)
    w_glob_keys = []
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    return net_glob, w_glob_keys
 

def get_preproc_model(args, dim_in, dim_out=2):
    #n_out = 2
    # 确保 dim_in 是有效的
    if dim_in <= 0:
        raise ValueError("dim_in must be a positive integer")
    net_preproc = MLPReg_preproc(dim_in, args.n_hidden, dim_out = dim_out)

    return net_preproc



#---------------------------------------------------------------------------
#
#               REGression
#
#---------------------------------------------------------------------------
class MLPReg_tensor_out(nn.Module):
    def __init__(self, dim_in,dim_out=1):
        super(MLPReg_tensor_out, self).__init__()
        self.layer_out = nn.Linear(dim_in,dim_out)
        self.weight_keys = [['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
 
        x = self.layer_out(x)
        return x 

class MLPReg_preproc(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLPReg_preproc, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.sigmoid = nn.LeakyReLU(0.2)
        self.layer_out = nn.Linear(dim_hidden,dim_out)


    def forward(self, x):
        x = self.layer_input(x)
        x = self.sigmoid(x)
        x = self.layer_out(x)
        x = self.sigmoid(x)
        return x

class MLPReg_HetNN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, drop=False):
        super(MLPReg_HetNN, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.sigmoid = nn.LeakyReLU(0.2)

        self.layer_hidden1 = nn.Linear(dim_hidden, dim_hidden)
        #self.layer_hidden2 = nn.Linear(dim_hidden, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden,1)
        self.weight_keys = [['layer_input.weight', 'layer_input.bias'],
                            ['layer_hidden1.weight', 'layer_hidden1.bias'],
                            #['layer_hidden2.weight', 'layer_hidden2.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.sigmoid(x)
        x = self.layer_hidden1(x)
        x_r = self.sigmoid(x)  # useful for hetarch (alignment on pre-last layer)
        x = self.layer_out(x_r)
        return x, x_r

