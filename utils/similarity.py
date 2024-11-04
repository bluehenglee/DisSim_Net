import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from PIL import Image, ImageDraw
from scipy import ndimage

def distance(x1,y1,x2,y2):
    # i越大距离大？
    sigma_i = 0.005
    # sigma_i = 0.025
    max_dis = np.sqrt(256**2 + 256 **2)
    dis = np.sqrt(np.abs((x1-x2)**2+(y1-y2)**2)) / max_dis
    if x1 == x2 and y1 == y2:
        dist = 1
    else:
        dist = np.exp((-(dis**2)/(2*sigma_i)))

    return dist

def similar(image,x1,y1,x2,y2):
    sigma_sim = 0.15
    sim = np.abs(image[int(x1)][int(y1)] - image[x2][y2])
    if x1 == x2 and y1 == y2:
        simi = 1
    else:
        simi = np.exp((-(sim) / (2*sigma_sim)))

    return simi

def save_dismap(image, kepoint,img_name):
    new_image = np.zeros(image.shape)
    # new_image = torch.zeros_like(image)
    for row in range(len(image)):
        for col in range(len(image[0])):

            n_y1 = int(kepoint[0,0])
            n_x1 = int(kepoint[0,1])
            n_y2 = int(kepoint[1,0])
            n_x2 = int(kepoint[1,1])
            n_y3 = int(kepoint[2,0])
            n_x3 = int(kepoint[2,1])
            n_y4 = int(kepoint[3,0])
            n_x4 = int(kepoint[3,1])
            # if n_x >= len(image):
            #     n_x -= len(image)
            # if n_y >= len(image[0]):
            #     n_y -= len(image[0])
            gs1 = distance(n_x1, n_y1, row, col)
            gi1 = similar(image,n_x1, n_y1, row, col)

            gs2 = distance(n_x2, n_y2, row, col)
            gi2 = similar(image,n_x2, n_y2, row, col)

            gs3 = distance(n_x3, n_y3, row, col)
            gi3 = similar(image,n_x3, n_y3, row, col)

            gs4 = distance(n_x4, n_y4, row, col)
            gi4 = similar(image,n_x4, n_y4, row, col)

            wp1 = gi1 * gs1 * 255
            wp2 = gi2 * gs2 * 255
            wp3 = gi3 * gs3 * 255
            wp4 = gi4 * gs4 * 255
            wp = wp1 + wp2 + wp3 + wp4

            new_image[row][col] = wp
    save_output = './dismap'
    if not os.path.exists(save_output):
        os.makedirs(save_output)
    save_img = os.path.join(save_output, img_name)
    cv2.imwrite(save_img, new_image)
    # return torch.tensor(new_image,dtype=torch.float32)