import keras
import numpy as np
import pandas as pd
import cv2
import os
import time
from sklearn.externals import joblib
import argparse

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
def read_data_test(path,dim):
    test_X = [] 
    test_files = []
    file_list = os.listdir(path) 
    for f in file_list:
        img = get_im_cv2(path + f)
        img = pre_process(img)
        test_X.append(img)
        f_name = f.split('_')[0]
        test_files.append(f_name)
    return np.array(test_X),test_files


# start the inference

def inference_test(test_X,model_save_dest,n_class):
    folds = len(list(model_save_dest.keys()))
    pred = np.zeros((len(test_X),n_class))
    for k in range(1,folds + 1):
        model = keras.models.load_model(model_save_dest[k])
        pred = pred + model.predict(test_X)
    pred = pred/(1.0*folds) 
    pred_class = np.argmax(pred,axis=1) 
    return pred_class 

def main(path,dim,model_save_dest,outdir,n_class):
    test_X,test_files = read_data_test(path,dim)
    pred_class = inference_test(test_X,model_save_dest,n_class)
    out = pd.DataFrame()
    out['id'] = test_files
    out['class'] = pred_class
    out['class'] = out['class'].apply(lambda x:'class' + str(x))
    out.to_csv(outdir + "results.csv",index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--path',help='path of images to run inference on')
    parser.add_argument('--dim',type=int,help='Image dimension size to process',default=224)
    parser.add_argument('--model_save_dest',help='location of the trained models')
    parser.add_argument('--n_class',type=int,help='No of classes')
    parser.add_argument('--outdir',help='Output DIrectory')
    args = parser.parse_args()
    path = args.path
    dim = args.dim
    model_save_dest = joblib.load(args.model_save_dest)
    n_class = args.n_class
    outdir = args.outdir
    main(path,dim,model_save_dest,outdir,n_class)





    
