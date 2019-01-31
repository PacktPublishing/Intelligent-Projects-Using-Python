# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:10:12 2018

@author: santanu
"""

import numpy as np
import os
from scipy.misc import imread
from scipy.misc import imsave
import fire
from elapsedtimer import ElapsedTimer
from pathlib import Path
import shutil        
'''
Process the images in Domain A and Domain and resize appropriately
Inputs contain the Domain A and Domain B image in the same image
This program will break them up and store them in their respecective folder

'''

def process_data(path,_dir_):
    os.chdir(path)
    try:         
        os.makedirs('trainA')
    except:
        print(f'Folder trainA already present, cleaning up  and recreating empty folder trainA')
        try:
            os.rmdir('trainA')
        except:
            shutil.rmtree('trainA')
           
        os.makedirs('trainA')

    try:         
        os.makedirs('trainB')
    except:
        print(f'Folder trainA already present, cleaning up  and recreating empty folder trainB')
        try:
            os.rmdir('trainB')
        except:
            shutil.rmtree('trainB')
        os.makedirs('trainB')
    path = Path(path)  
    files = os.listdir(path /_dir_)
    print('Images to process:', len(files))
    i = 0
    for f in files:
        i+=1 
        img = imread(path / _dir_ / str(f))
        w,h,d = img.shape
        h_ = int(h/2)
        img_A = img[:,:h_]
        img_B = img[:,h_:]
        imsave(f'{path}/trainA/{str(f)}_A.jpg',img_A)
        imsave(f'{path}/trainB/{str(f)}_B.jpg',img_A)
        if ((i % 10000) == 0 & (i >= 10000)):
            print(f'the number of input images processed : {i}')
    files_A = os.listdir(path / 'trainA')
    files_B = os.listdir(path / 'trainB')
    print(f'No of images written to {path}/trainA is {len(files_A)}')
    print(f'No of images written to {path}/trainA is {len(files_B)}')
        
        
with ElapsedTimer('process Domain A and Domain B Images'):
    fire.Fire(process_data)



