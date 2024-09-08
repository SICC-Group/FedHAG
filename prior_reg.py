#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:07:39 2022

@author: alain
"""

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# Regularizer used in proximal-relational autoencoder
class Prior(nn.Module):
    def __init__(self, z_dim, total_mean_train):
        super(Prior, self).__init__()
        self.z_dim = z_dim
        self.mu_init = total_mean_train
        self.mu = self.mu_init.clone()
        #self.logvar = torch.ones(data_size)
        self.logvar = torch.zeros(z_dim)
        self.mu_temp = torch.zeros(z_dim)
        self.n_update = 0
        self.mu_local = None

    def forward(self):
        return self.mu, self.logvar

    def sampling_gaussian(self, num_sample, mean, logvar):
        num_sample = num_sample # =batch_size
        self.z_dim = mean.shape # =dim_latent
        std = torch.exp(0.5 * logvar)
        samples = mean + torch.randn(num_sample, self.z_dim[0]) * std
        return samples
    
    # def sampling_gmm(self,num_sample):
    #     std = torch.exp(0.5 * self.logvar)
    #     n = int(num_sample / self.mu.size(0)) + 1
    #     for i in range(n):
    #         eps = torch.randn_like(std)
    #         if i == 0:
    #             samples = self.mu + eps * std
    #         else:
    #             samples = torch.cat((samples, self.mu + eps * std), dim=0)
    #     return samples[:num_sample, :]
    
    def sampling_gmm(self, num_sample):
    # 计算标准差
        std = torch.exp(0.5 * self.logvar)
        # 从高斯分布中采样
        eps = torch.randn((num_sample, self.z_dim))
        samples = self.mu.unsqueeze(0).repeat(num_sample, 1) + eps * std.unsqueeze(0)
        return samples
    
    def init_mu_temp(self):
        self.mu_temp = torch.zeros(self.z_dim)
        self.n_update = 0


# Simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Initialize the Prior
    prior = Prior(data_size=[2, 1])

    # Sample from the prior for regression
    num_samples = 100
    x_samples = prior.sampling_gaussian(num_samples, prior.mu[0], prior.logvar[0])

    # Generate true regression parameters
    true_weights = torch.tensor([[2.0]])
    true_bias = torch.tensor([1.0])
    y_samples = x_samples @ true_weights + true_bias + 0.1 * torch.randn_like(x_samples)  # Adding noise

    # Define and train a simple linear regression model
    model = LinearRegression(input_dim=1, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Prepare data
    x_train = x_samples
    y_train = y_samples

    # Training loop
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

    # Plot the results
    model.eval()
    with torch.no_grad():
        y_pred = model(x_train)

    plt.scatter(x_train.numpy(), y_train.numpy(), color='blue', label='True Data')
    plt.plot(x_train.numpy(), y_pred.numpy(), color='red', label='Fitted Line')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression with Samples from Prior')
    plt.show()
