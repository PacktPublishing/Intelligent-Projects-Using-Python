import keras
import numpy as np
import pandas as pd
import cv2
import os
import time

# Read the Image and resize to the suitable dimension size
def get_im_cv2(path,dim=224):
    img = cv2.imread(path)
    resized = cv2.resize(img, (dim,dim), cv2.INTER_LINEAR)
    return resized

# Pre Process the Images based on the ImageNet pre-trained model Image transformation
def pre_process(img):
    img[:,:,0] = img[:,:,0] - 103.939
    img[:,:,1] = img[:,:,0] - 116.779
    img[:,:,2] = img[:,:,0] - 123.68
    return img

   
# Function to build test input data
def read_data_test(class_folders,path,dim,train_val='train'):
    test_X = [] 
    test_files = []
    file_list = os.listdir(path) 
    for f in file_list:
        img = get_im_cv2(path + '/' + f)
        img = pre_process(img)
        test_X.append(img)
        f_name = f.split('_')[0]
        test_files.append(f_name)
    return np.array(test_X),test_files


# start the inference

def inference_test(test_X,test_files,model_save_dest,n_class=5,folds=5):
    pred = np.zeros((len(test_X),n_class))
    for k in xrange(1,folds + 1):
        model = keras.models.load_model(model_save_dest[k])
        pred = pred + model.predict(test_X)
    pred = pred/(1.0*folds) 
    pred_class = np.argmax(pred,axis=1) 
    return pred_class 

if __name__ == "__main__":
    start_time = time.time()
    path = '/home/santanu/Downloads/Diabetic Retinopathy/New/'
    class_folders = ['0','1','2','3','4']
    num_class = len(class_folders)
    dim = 224
    lr = 1e-5
    print 'Starting time:',start_time
    test_X,test_files = read_data_test(class_folders,path,num_class,dim)
    dest = path + "dict_model"
    model_save_dest = np.load(dest)
    pred_class = inference_test(test_X,test_files,model_save_dest,n_class=5,folds=5)
    temp = pd.DatFrame()
    temp['id'] = test_files
    temp['class'] = pred_class
    id_unique = np.unique(test_files)
    class_merged = []
    for id_ in id_unique:
        class_ = np.max(temp[temp['id'] == id_]['class'].values,axis=0)
        class_merged.append(class_)
    out = pd.DataFrame()
    out['id'] = id_unique
    out['class'] = class_merged 
    out.to_csv(path + "results.csv",index=False)
