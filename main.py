#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 23:13:41 2024

@author: wangxinping
"""
import os
import cv2
import numpy as np
import torch
from functools import partial

import torch.nn as nn
import torchvision.transforms.v2 as tsfm_v2

from lib.models.conv import vgg16_net
from lib.models.deconv import vgg16_deconvnet
from lib.utils import image_transformation, prediction


tsfm = tsfm_v2.Compose([
    tsfm_v2.ToImage(),
    tsfm_v2.ToDtype(torch.float32, scale=True)
    ])

def register_model(net):
    
    def hook(module, input, output, layer_idx):
        if isinstance(module, nn.MaxPool2d):
            net.feature_maps[layer_idx] = output[0]
            net.pooling_indices[layer_idx] = output[1]
        if isinstance(module, nn.Conv2d):
            net.feature_maps[layer_idx] = output
            
    for layer_idx, layer in enumerate(net.features):
        layer.register_forward_hook(partial(hook, layer_idx=layer_idx))

# only keep one layer with maximum value 
def vis_layer(net, deconvnet, layer_idx):
    feature_maps = net.feature_maps[layer_idx]
    
    each_layer_max_value, _ = feature_maps.flatten(-2).max(-1)
    max_value, best_map_idx = each_layer_max_value.max(-1)
    max_value, best_map_idx = max_value[0], best_map_idx[0]
    
    best_map = feature_maps[:,  best_map_idx, :, :]
    tsfm_best_map = torch.where(best_map == max_value, best_map, torch.zeros(size=best_map.size()))
    
    tsfm_feature_maps = torch.zeros(size=feature_maps.size())
    tsfm_feature_maps[:, best_map_idx, :, :] = tsfm_best_map
    deconv_output = deconvnet(tsfm_feature_maps, layer_idx, net.pooling_indices)
    
    result = torch.squeeze(deconv_output).permute(1, 2, 0).data.numpy()
    result = (result - result.min()) / (result.max() - result.min()) * 255
    feature_img = result.astype(np.uint8).copy()
    
    return feature_img

# keep all layer with maximum value
def vis_layer_v2(net, deconvnet, layer_idx):
    feature_maps = net.feature_maps[layer_idx]
    
    each_layer_max_value, _ = feature_maps.flatten(-2).max(-1)
    each_layer_max_value = torch.squeeze(each_layer_max_value).data.numpy()
    
    tsfm_feature_maps = torch.zeros(size=feature_maps.size())
    for idx, max_value in enumerate(each_layer_max_value):
        feat_map = feature_maps[:, idx, :, :]
        tsfm_map = torch.where(feat_map==max_value, feat_map, torch.zeros(size=feat_map.size()))
        tsfm_feature_maps[:, idx, :, :] = tsfm_map
    deconv_output = deconvnet(tsfm_feature_maps, layer_idx, net.pooling_indices)
    
    result = torch.squeeze(deconv_output).permute(1, 2, 0).data.numpy()
    result = (result - result.min()) / (result.max() - result.min()) * 255
    feature_img = result.astype(np.uint8).copy()
    
    return feature_img

        
if __name__ == "__main__":
    root_dir = "/Users/wangxinping/Desktop/github/Visualizing_CNN/data"
    filename = "maltese.jpg"
    
    img = cv2.imread(os.path.join(root_dir, filename))
    resized_img, transformed_img = image_transformation(img, tsfm=tsfm)

    C, H, W = transformed_img.size()
    transformed_img = transformed_img.view(1, C, H, W)

    net = vgg16_net()
    register_model(net)
    
    prob = net(transformed_img)
    prediction(prob)
    
    vis_layer_indicies = [14, 17, 19, 21, 24, 26, 28]
    deconvnet = vgg16_deconvnet()
    
    row1 = resized_img.copy()
    for idx, layer_idx in enumerate(vis_layer_indicies):
        deconv_result_img = vis_layer_v2(net, deconvnet, layer_idx)
        cv2.putText(deconv_result_img, f"layer{layer_idx}", (10, 210), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
        if idx <=2:
            row1 = np.hstack((row1, deconv_result_img))
            
        elif idx == 3:
            row2 = deconv_result_img
            
        else:
            row2 = np.hstack((row2, deconv_result_img))
            
    all_images = np.vstack((row1, row2))
            
    cv2.imshow("img", all_images)
    cv2.waitKey(0)    
    cv2.destroyAllWindows()
    

        

    
    
