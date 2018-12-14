# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 08:33:48 2018

@author: santanu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = '/home/santanu/Downloads/Mobile_App/aclImdb/processed_file.csv'

def length_function(text):
    words = text.split()
    return len(words)
     
df = pd.read_csv(path)

df['length'] = df['review'].apply(length_function)		
x = df['length'].values
plt.hist(x)

np.sum(x[x <= 1000])/np.float(np.sum(x))
 




