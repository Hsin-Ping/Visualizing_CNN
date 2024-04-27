#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:14:53 2024

@author: wangxinping
"""

import torch.nn as nn
from torchvision.models import vgg16

class vgg16_deconvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.MaxUnpool2d(2, 2), #(batch size, 512, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            
            nn.MaxUnpool2d(2, 2), #(batch size, 256, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            
            nn.MaxUnpool2d(2, 2), #(batch size, 128, 56, 56)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            
            nn.MaxUnpool2d(2, 2), #(batch size, 64, 112, 112)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            
            nn.MaxUnpool2d(2, 2),  #(batch size, 64, 224, 224)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, 1),
            )
        self.get_corresponding_layers()
        self.load_pretraiend_weights()
        return
    
    def get_corresponding_layers(self):
        self.net = vgg16(weights="DEFAULT")
        conv2d_indicies = [layer_idx for layer_idx, layer in enumerate(self.net.features) if isinstance(layer, nn.Conv2d)]
        deconv2d_indicies = [layer_idx for layer_idx, layer in enumerate(self.features) if isinstance(layer, nn.ConvTranspose2d)]
        
        pooling_indicies = [layer_idx for layer_idx, layer in enumerate(self.net.features) if isinstance(layer, nn.MaxPool2d)]
        unpooling_indicies = [layer_idx for layer_idx, layer in enumerate(self.features) if isinstance(layer, nn.MaxUnpool2d)]
        
        deconv2d_indicies.reverse()
        pooling_indicies.reverse()
        
        self.conv2deconv = dict([(conv_idx, deconv_idx) for conv_idx, deconv_idx in zip(conv2d_indicies, deconv2d_indicies)])
        self.unpooling2pooling = dict([(unpool_idx, pool_idx) for unpool_idx, pool_idx in zip(unpooling_indicies, pooling_indicies)])
    
    def load_pretraiend_weights(self):
        for layer_idx, layer in enumerate(self.net.features):
            if layer_idx in self.conv2deconv:
                self.features[self.conv2deconv[layer_idx]].weight.data = layer.weight.data
                #self.features[self.conv2deconv[layer_idx]].bias.data = layer.bias.data # will have error show up
                
    
    def forward(self, x, layer_idx, pooling_indicies):
        deconv_init_layer = self.conv2deconv[layer_idx]
        for deconv_layer_idx in range(deconv_init_layer, len(self.features)):
            layer = self.features[deconv_layer_idx]
            if isinstance(layer, nn.MaxUnpool2d):
                x = layer(x, pooling_indicies[self.unpooling2pooling[deconv_layer_idx]])
            else:
                x = layer(x)
        return x