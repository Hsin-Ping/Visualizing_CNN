#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:38:37 2024

@author: wangxinping
"""

import torch
import torch.nn as nn
from torchvision.models import vgg16

class deconVgg16_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # block1
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, padding=1),
            
            # block2
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 3, 1, padding=1),
            
            # block3
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 1, padding=1),
            
            # block4
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 1, padding=1),
            
            # block5
            nn.MaxUnpool2d(2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, 1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, 1, padding=1),
            )
        
        
        self.get_coressponding_layers()
        #self.get_layer_corresponding_indices()
        self.load_pretrained_weights()
        #self.init_weight()
        return
    
    def get_layer_corresponding_indices(self):
        conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
        pooling_indices = [4, 9, 16, 23, 30]
        deconv_indices = [2, 4, 6, 9, 11, 13, 16, 18, 20, 23, 25, 28, 30]
        unpooling_indices = [0, 7, 14, 21, 26]
        conv_indices.reverse()
        pooling_indices.reverse()
        conv2deconv_dict = [(conv_idx, deconv_idx) for conv_idx, deconv_idx in zip(conv_indices, deconv_indices)]
        unpool2pool_dict = [(unpooling_idx, pooling_idx) for pooling_idx, unpooling_idx in zip(pooling_indices, unpooling_indices)]
        self.conv_to_deconv = dict(conv2deconv_dict)
        self.unpooling_to_pooling  = dict(unpool2pool_dict)
    
    def get_coressponding_layers(self):
        net = vgg16()
        conv2d_indices = []
        pooling_indices = []
        deconv2d_indices = []
        unpooling_indices = []
        

        for idx, layer in enumerate(net.features):
            if isinstance(layer, nn.Conv2d):
                conv2d_indices.append(idx)
                
        for idx, layer in enumerate(net.features):
            if isinstance(layer, nn.MaxPool2d):
               pooling_indices.append(idx) 
               
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.ConvTranspose2d):
                deconv2d_indices.append(idx)
                
        for idx, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxUnpool2d):
               unpooling_indices.append(idx)

        deconv2d_indices.reverse()
        pooling_indices.reverse()
        
        
        self.conv_to_deconv = dict([(conv_idx, deconv_idx) for conv_idx, deconv_idx in zip(conv2d_indices, deconv2d_indices)])
        self.unpooling_to_pooling = dict([(unpool_idx, pool_idx) for unpool_idx, pool_idx in zip(unpooling_indices, pooling_indices)])
        print(self.conv_to_deconv)
        print(self.unpooling_to_pooling)
        self.conv2deconv_dict = self.conv_to_deconv
        self.unpool2pool_dict = self.unpooling_to_pooling
    
    def load_pretrained_weights(self):
        net = vgg16(pretrained=True)
        for idx, layer in enumerate((net.features)):
            if idx in self.conv_to_deconv and isinstance(layer, nn.Conv2d):
                print("here", idx)
                self.features[self.conv_to_deconv[idx]].weight.data = net.features[idx].weight.data
                #self.features[self.conv_to_deconv[idx]].bias.data = net.features[idx].bias.data
    """
    def init_weight(self):
        vgg16Conv = vgg16(pretrained=True)
        for idx, layer in enumerate(vgg16Conv.features):
            if isinstance(layer, nn.Conv2d):
               self.features[self.conv2deconv_dict[idx]].weight.data = layer.weight.data
               #self.features[self.conv2deconv_dict[idx]].bias.data = layer.bias.data
        return
    """
    def forward(self, x, layer_idx, pooling_loc):
        deconv_start_layer = self.conv_to_deconv[layer_idx]
        for idx in range(deconv_start_layer, len(self.features)):
            if idx in self.unpooling_to_pooling:
                x = self.features[idx](x, pooling_loc[self.unpooling_to_pooling[idx]])
            else:
                x = self.features[idx](x)
        return x

if __name__ == "__main__":
    denet = deconVgg16_net()
    