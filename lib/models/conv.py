#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:14:01 2024

@author: wangxinping
"""

import torch.nn as nn
from torchvision.models import vgg16

class vgg16_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True), #(batch size, 64, 112, 112)
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True), #(batch size, 128, 56, 56)
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True), #(batch size, 256, 28, 28)
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True), #(batch size, 512, 14, 14)
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, return_indices=True), #(batch size, 512, 7, 7)
            )
        """
        nn.flatten(start_dim, end_dim)
        - start_dim (int) – first dim to flatten (default = 1).
        - end_dim (int) – last dim to flatten (default = -1).
        """
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1)
            )
        
        self.load_pretraiend_weights()
        self.feature_maps = dict()
        self.pooling_indices = dict()
        return
    
    def load_pretraiend_weights(self):
        net = vgg16(weights="DEFAULT")
        for idx, layer in enumerate(net.features):
            if isinstance(layer, nn.Conv2d):
                self.features[idx].weight.data = layer.weight.data
                self.features[idx].bias.data = layer.bias.data
                
        for idx, layer in enumerate(net.classifier):
            if isinstance(layer, nn.Linear):
                self.classifier[idx].weight.data = layer.weight.data
                self.classifier[idx].bias.data = layer.bias.data  
        return
    
    def forward(self, x):
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                x, indicies = layer(x)
            else:
                x = layer(x)

        out = self.flatten(x)
        prob = self.classifier(out)
        return prob
    
if __name__ == "__main__":
    net = vgg16(weights="DEFAULT")