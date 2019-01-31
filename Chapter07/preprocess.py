# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 22:36:00 2018
@author: santanu
"""
import numpy as np
import pandas as pd
import os
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import fire 
from elapsedtimer import ElapsedTimer



#path = '/home/santanu/Downloads/Mobile_App/aclImdb/'

# Function to clean the text and convert it into lower case    
def text_clean(text):
    letters = re.sub("[^a-zA-z0-9\s]", " ",text)
    words = letters.lower().split()
    text = " ".join(words)
    
    return text


def process_train(path):
    review_dest = []
    reviews = []
    train_review_files_pos = os.listdir(path + 'train/pos/')
    review_dest.append(path + 'train/pos/')
    train_review_files_neg = os.listdir(path + 'train/neg/')
    review_dest.append(path + 'train/neg/')
    test_review_files_pos = os.listdir(path + 'test/pos/') 
    review_dest.append(path + 'test/pos/')
    test_review_files_neg = os.listdir(path + 'test/neg/')
    review_dest.append(path + 'test/neg/')
    
    sentiment_label = [1]*len(train_review_files_pos) +  \
                      [0]*len(train_review_files_neg) + \
                      [1]*len(test_review_files_pos) + \
                      [0]*len(test_review_files_neg)
                      
    review_train_test = ['train']*len(train_review_files_pos) + \
                        ['train']*len(train_review_files_neg) + \
                        ['test']*len(test_review_files_pos) + \
                        ['test']*len(test_review_files_neg)
    
    reviews_count = 0     
    
    for dest in review_dest:
        files = os.listdir(dest)
        for f in files:
            fl = open(dest + f,'r')
            review = fl.readlines()
            review_clean = text_clean(review[0])
            reviews.append(review_clean)
            reviews_count +=1
            
    df  = pd.DataFrame()
    df['Train_test_ind'] = review_train_test
    df['review'] = reviews
    df['sentiment_label'] = sentiment_label
    df.to_csv(path + 'processed_file.csv',index=False)
    print ('records_processed',reviews_count)
    return df
    
def process_main(path):
    df = process_train(path)
    # We will tokenize the text for the most common 50000 words.
    max_fatures = 50000
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(df['review'].values)
    X = tokenizer.texts_to_sequences(df['review'].values) 
    X_ = []
    for x in X:
        x = x[:1000]    
        X_.append(x)   
    X_ = pad_sequences(X_)
    y = df['sentiment_label'].values
    index = list(range(X_.shape[0]))
    np.random.shuffle(index)
    train_record_count = int(len(index)*0.7)
    validation_record_count = int(len(index)*0.15)

    train_indices = index[:train_record_count]
    validation_indices = index[train_record_count:train_record_count + 
                              validation_record_count]
    test_indices  = index[train_record_count + validation_record_count:]
    X_train,y_train =     X_[train_indices],y[train_indices]
    X_val,y_val     =    X_[validation_indices],y[validation_indices]
    X_test,y_test =     X_[test_indices],y[test_indices]
  
    np.save(path + 'X_train',X_train)
    np.save(path + 'y_train',y_train)
    np.save(path + 'X_val',X_val)
    np.save(path + 'y_val',y_val)
    np.save(path + 'X_test',X_test)
    np.save(path + 'y_test',y_test)
    
    

    # saving the tokenizer oject for inference
    with open(path + 'tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    with ElapsedTimer('Process'):
        fire.Fire(process_main)







    
    

