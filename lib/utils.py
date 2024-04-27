#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:18:40 2024

@author: wangxinping
"""
import cv2
import json
import urllib.request as request
import torch

def image_transformation(img, resize=(224,224), tsfm=None):
    resized_img = cv2.resize(img, resize)
    
    if tsfm:
        transformed_img = tsfm(resized_img)
        return resized_img, transformed_img
    else:
        return resized_img
    
def prediction(prob):
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    labels = json.load(request.urlopen(url))
    # torch.topk: If dim is not given, the last dimension of the input is chosen.
    top5_k, top5_indices = torch.topk(prob, 5)
    top5_k, top5_indices  = top5_k[0].data.numpy(),  top5_indices[0].data.numpy()
    top5_labels = dict([(labels[str(idx)][1], value) for value, idx in zip(top5_k, top5_indices)])
    print("prediction result", end="\n")
    print(top5_labels)