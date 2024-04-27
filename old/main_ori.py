#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:18:50 2024

@author: wangxinping
"""
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from functools import partial

from models.VGG import vgg16_net
from models.deconVGG import deconVgg16_net, deconVgg16_net_ori
from utils.utils import prediction



def get_transformed_image(image_path):
    tsfm = transforms.Compose(
        [transforms.ToTensor()]
        )
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    transformed_image = tsfm(img)
    transformed_image  = transformed_image.unsqueeze_(0)
    return img, transformed_image

def store(model):
    
    def hook(module, input, output, layer_idx):
        if isinstance(module, nn.MaxPool2d):
            model.feature_maps[layer_idx] = output[0]
            model.pooling_loc[layer_idx] = output[1]
        elif isinstance(module, nn.Conv2d):
            model.feature_maps[layer_idx] = output
            
    for idx, layer in enumerate(model.features):
        layer.register_forward_hook(partial(hook, layer_idx=idx))
        
def vis_layer(layer, vgg16conv, vgg16deconv, feature_maps, maxpooling_loc):
    
    layer_feature_maps = feature_maps[layer]
    print("layer_feature_maps",layer_feature_maps.size())
    
    max_activations = []
    for idx in range(0, layer_feature_maps.size()[1]):
        feature_map = layer_feature_maps[:,idx,:,:]
        max_activations.append(torch.max(feature_map))
    max_activation = max(max_activations)
    #print(max_activations)

    choose_map_idx = max_activations.index(max_activation)
    choose_map = layer_feature_maps[:, choose_map_idx, :,:]
    #print(choose_map.size())
    tsfm_layer_features = torch.zeros(layer_feature_maps.size())
    #print(tsfm_layer_features.size())
    tsfm_choose_map = torch.where(choose_map==max_activation, choose_map, torch.zeros(size=choose_map.size()))
    tsfm_layer_features[:, choose_map_idx, :, :] = tsfm_choose_map
    #print("tsfm_layer_features",tsfm_layer_features.size())
    deconv_output = vgg16deconv(tsfm_layer_features, layer, vgg16conv.maxpooling_loc)
    
    feature_img = deconv_output.data.numpy()[0].transpose(1,2,0)
    feature_img = (feature_img-feature_img.min()) / (feature_img.max()  - feature_img.min()) * 255
    # have to add copy() to put text on image
    feature_img = feature_img.astype(np.uint8).copy()
    
    return feature_img

def vis_layer2(layer, vgg16conv, vgg16deconv, feature_maps, maxpooling_loc):
    
    layer_feature_maps = feature_maps[layer]
    print("layer_feature_maps",layer_feature_maps.size())
    
    tsfm_layer_features = layer_feature_maps.clone()
    for idx in range(0, layer_feature_maps.size()[1]):
        feature_map = layer_feature_maps[:,idx,:,:]
        tsfm_feature = torch.where(feature_map==torch.max(feature_map), feature_map, torch.zeros(size=feature_map.size()))
        tsfm_layer_features[:, idx, :, :] = tsfm_feature
        #max_activations.append(torch.max(feature_map))
        
    #max_activation = max(max_activations)
    #print(max_activations)

    #tsfm_layer_features = np.array()
    
    #choose_map_idx = max_activations.index(max_activation)
    #choose_map = layer_feature_maps[:, choose_map_idx, :,:]
    #print(choose_map.size())
    #tsfm_layer_features = torch.zeros(layer_feature_maps.size())
    #print(tsfm_layer_features.size())
    #tsfm_choose_map = torch.where(choose_map==max_activation, choose_map, torch.zeros(size=choose_map.size()))
    #tsfm_layer_features[:, choose_map_idx, :, :] = tsfm_choose_map
    #print("tsfm_layer_features",tsfm_layer_features.size())
    deconv_output = vgg16deconv(tsfm_layer_features, layer, vgg16conv.pooling_loc)
    
    feature_img = deconv_output.data.numpy()[0].transpose(1,2,0)
    feature_img = (feature_img-feature_img.min()) / (feature_img.max()  - feature_img.min()) * 255
    # have to add copy() to put text on image
    feature_img = feature_img.astype(np.uint8).copy()
    
    return feature_img
    
        

        

if __name__ == "__main__":
    
    image_path = "./data/cat.jpg"
    
    resize_img, transformed_image = get_transformed_image(image_path)
    net = vgg16_net()
    store(net)
    #pred = net(transformed_image)
    #print("pooling_loc",net.maxpooling_loc[30].size())
    top5_classes = prediction(transformed_image, net)
    print(top5_classes)
    
    vgg16deconv = deconVgg16_net_ori()
    
    vis_layers = [14, 17, 19, 21, 24, 26, 28]
    
    all_images = resize_img.copy()
    
    for idx, layer in enumerate(vis_layers):
        print("idx",idx)
        feature_img = vis_layer2(layer, net, vgg16deconv, net.feature_maps, net.pooling_loc)
        cv2.putText(feature_img, f"layer{layer}", (160,210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), thickness=2)
        if idx == 3:
            col2 = feature_img
        elif idx > 3:
            col2 = np.hstack((col2, feature_img))
        else:
            all_images = np.hstack((all_images,feature_img))
    all_images = np.vstack((all_images, col2))
        
    
    cv2.imshow("feature_img", all_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
        
    