# -*- coding: utf-8 -*-
"""Markov_Random_Fields.ipynb

"""

import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import numpy as np

"""
Get Segmented Image from output of MRF
"""
def get_segmented_image(set1,set2,img_prob,min_i,min_j):
    seg_img=np.zeros(shape=img_prob.shape)
    height=img_prob.shape[1]
    width=img_prob.shape[0]
    for i in range(height):
        for j in range(width):
            if (i,j) in set1:
                seg_img[min_i+i][min_j+j]=255
    return seg_img
"""
Get Window of Image
"""
def find_window(img_prob):
    max_ind=img_prob.shape[0]
    indices=np.argwhere(img_prob<0.5)
    if len(indices)==0:
        return 0,0,img_prob.shape[0],img_prob.shape[1]
    min_i = 0 if np.min(indices[:,0]) <= 10 else np.min(indices[:,0])-10
    min_j = 0 if np.min(indices[:,1]) <= 10 else np.min(indices[:,1])-10
    max_i = max_ind-1 if np.max(indices[:,0]) >= max_ind-10 else np.max(indices[:,0])+10
    max_j = max_ind-1 if np.max(indices[:,1]) >=max_ind-10 else np.max(indices[:,1])+10
    return min_i,min_j,max_i,max_j
"""
Create MRF Graph
"""
def create_markov_field(img_prob,img_dep):
    min_i,min_j,max_i,max_j=find_window(img_prob)
    height=max_i-min_i
    width=max_j-min_j
    G=nx.grid_2d_graph(height,width)
    for i in range(0, height):
        for j in range(0, width-1):
            G[(i,j)][(i,j+1)]['weight'] =.1

    for i in range(0, height-1):
        for j in range(0, width):
            G[(i,j)][(i+1,j)]['weight'] = .1
    
    depth_diff=np.max(img_prob[min_i:max_i,min_j:max_j])-np.min(img_prob[min_i:max_i,min_j:max_j])
    depth_sd=np.std(img_prob[min_i:max_i,min_j:max_j])
    depth_mean=np.mean(img_prob[min_i:max_i,min_j:max_j])
    if (depth_mean<0.45 and depth_diff<0.95) or depth_sd <0.2 :
        magic=10
    else:
        magic=30
    for i in range(0, height):
        for j in range(0, width):
            de=img_dep[min_i+i][min_j+j]/magic
            ac=img_prob[min_i+i][min_j+j]
            G.add_edge('t', (i,j), weight=ac)
            G.add_edge('s', (i,j), weight=de)
    return depth_diff,depth_sd,depth_mean,magic,G
"""
Add Depth Information to improve Segmentation
"""
def improve_segmentation_with_depth(img_prob,dep_img):
    norm_dep_img=(dep_img-np.min(dep_img))/(np.max(dep_img)-np.min(dep_img))
    img_dep_inv=1/(norm_dep_img+1e-9)
    img_prob_inv=1-img_prob
    depth_diff,depth_sd,depth_mean,magic,G=create_markov_field(img_prob_inv,img_dep_inv)
    (cost, (set1, set2)) = nx.minimum_cut(G, 's', 't', capacity='weight')
    min_i,min_j,max_i,max_j = find_window(img_prob_inv)
    if len(set1)/(len(set2)+len(set1)) >  0.98 or len(set1)/(len(set2)+len(set1))  < 0.02:
        p=np.copy(img_prob)
        p[p>0.5]=255
        p[p<=5]=0
        return p
    return get_segmented_image(set1,set2,img_prob,min_i,min_j)
