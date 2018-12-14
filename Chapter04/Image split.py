# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:10:12 2018

@author: santanu
"""

import numpy as np
import os 
from scipy.misc import imread
from scipy.misc import imsave

import matplotlib.pyplot as plt

def process_data(path):
    os.chdir(path)
             
    os.makedirs('trainA')
    os.makedirs('trainB')
    files = os.listdir(path)
    print 'Images to process:', len(files)
    i = 0
    for f in files:
        i+=1 
        img = imread(path + str(f))
        w,h,d = img.shape
        img_A = img[:w,:h/2,:d]
        img_B = img[:w,h/2:h,:d]
        imsave(path + 'trainA/' + str(f) + '_A.jpg',img_A)
        imsave(path + 'trainB/' + str(f) + '_B.jpg',img_B)
        if ((i % 10000) == 0 & (i >= 10000)):
            print i
        
        
path = '/home/santanu/Downloads/DiscoGAN/edges2handbags/train/'
process_data(path)

files_A = os.listdir(path+ 'trainA')
len(files_A)
files_B = os.listdir(path+ 'trainB')
len(files_B)
