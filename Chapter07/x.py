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

