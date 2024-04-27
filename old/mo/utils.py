#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:48:54 2024

@author: wangxinping
"""
import torch
import urllib.request as request
import json


def prediction(img, model):
    url = " https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    label_dict = json.load(request.urlopen(url))
    prob = model(img)
    top5_prob, top5_indices = torch.topk(prob, 5)
    top5_prob, top5_indices = top5_prob.view(5).detach().numpy().tolist(), top5_indices.view(5).detach().numpy().tolist()
    result_dict = [(label_dict[str(idx)][1], round(prob,3)) for prob, idx in zip(top5_prob, top5_indices)]
    print(result_dict)
    #print(result_dict)

#url = " https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
#label_dict = json.load(request.urlopen(url))
#print(label_dict)

#if __name__ == "__main__":
    