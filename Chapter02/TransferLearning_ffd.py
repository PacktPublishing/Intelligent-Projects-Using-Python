_author__ = 'Santanu Pattanayak'

import numpy as np
np.random.seed(1000)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import GlobalMaxPooling2D,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.metrics import log_loss
import keras
from keras import __version__ as keras_version
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers 
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import h5py
import argparse
from sklearn.externals import joblib
import json
import keras 
from pathlib import Path
import glob

def pre_process(img):
    img[:,:,0] = img[:,:,0] - 103.939
    img[:,:,1] = img[:,:,0] - 116.779
    img[:,:,2] = img[:,:,0] - 123.68
    return img

class TransferLearning:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Process the inputs')
        parser.add_argument('--path',help='image directory')
        parser.add_argument('--class_folders',help='class images folder names')
        parser.add_argument('--dim',type=int,help='Image dimensions to process')
        parser.add_argument('--lr',type=float,help='learning rate',default=1e-4)
        parser.add_argument('--batch_size',type=int,help='batch size')
        parser.add_argument('--epochs',type=int,help='no of epochs to train')
        parser.add_argument('--initial_layers_to_freeze',type=int,help='the initial layers to freeze')
        parser.add_argument('--model',help='Standard Model to load',default='InceptionV3')
        parser.add_argument('--folds',type=int,help='num of cross validation folds',default=5)
        parser.add_argument('--outdir',help='output directory')
        args = parser.parse_args()
        self.path = Path(args.path)
        self.train_dir = Path(f'{self.path}/train/')
        self.val_dir = Path(f'{self.path}/validation/')
        self.class_folders = json.loads(args.class_folders)
        self.dim  = int(args.dim)
        self.lr   = float(args.lr)
        self.batch_size = int(args.batch_size)
        self.epochs =  int(args.epochs)
        self.initial_layers_to_freeze = int(args.initial_layers_to_freeze)
        self.model = args.model
        self.folds = int(args.folds)
        self.outdir = Path(args.outdir)

    def inception_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = InceptionV3(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final

# ResNet50 Model for transfer Learning 
    def resnet_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = ResNet50(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final

# VGG16 Model for transfer Learning 

    def VGG16_pseudo(self,dim=224,freeze_layers=10,full_freeze='N'):
        model = VGG16(weights='imagenet',include_top=False)
        x = model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        out = Dense(5,activation='softmax')(x)
        model_final = Model(input = model.input,outputs=out)
        if full_freeze != 'N':
            for layer in model.layers[0:freeze_layers]:
                layer.trainable = False
        return model_final


    def train_model(self,train_dir,val_dir,n_fold=5,batch_size=16,epochs=40,dim=224,lr=1e-5,model='ResNet50'):
        if model == 'Resnet50':
            model_final = self.resnet_pseudo(dim=224,freeze_layers=10,full_freeze='N')
        if model == 'VGG16':
            model_final = self.VGG16_pseudo(dim=224,freeze_layers=10,full_freeze='N') 
        if model == 'InceptionV3':
            model_final = self.inception_pseudo(dim=224,freeze_layers=10,full_freeze='N')
            
        train_file_names = glob.glob(f'{train_dir}/*/*')
        val_file_names = glob.glob(f'{val_dir}/*/*')
        train_steps_per_epoch = len(train_file_names)/float(batch_size)
        val_steps_per_epoch = len(val_file_names)/float(batch_size)
        train_datagen = ImageDataGenerator(horizontal_flip = True,vertical_flip = True,width_shift_range = 0.1,height_shift_range = 0.1,
                channel_shift_range=0,zoom_range = 0.2,rotation_range = 20,preprocessing_function=pre_process)
        val_datagen = ImageDataGenerator(preprocessing_function=pre_process)
        train_generator = train_datagen.flow_from_directory(train_dir,
        target_size=(dim,dim),
        batch_size=batch_size,
        class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(val_dir,
        target_size=(dim,dim),
        batch_size=batch_size,
        class_mode='categorical')
        print(train_generator.class_indices)
        joblib.dump(train_generator.class_indices,f'{self.outdir}/class_indices.pkl')
        adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model_final.compile(optimizer=adam, loss=["categorical_crossentropy"],metrics=['accuracy'])
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,patience=3, min_lr=0.000001)
        early = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
        logger = CSVLogger(f'{self.outdir}/keras-epochs_ib.log', separator=',', append=False)
        model_name = f'{self.outdir}/keras_transfer_learning-run.check'
        checkpoint = ModelCheckpoint(
                model_name,
                monitor='val_loss', mode='min',
                save_best_only=True,
                verbose=1) 
        callbacks = [reduce_lr,early,checkpoint,logger]
        model_final.fit_generator(train_generator,steps_per_epoch=train_steps_per_epoch,epochs=epochs,verbose=1,validation_data=(val_generator),validation_steps=val_steps_per_epoch,callbacks=callbacks,
                                                                                                                  class_weight={0:0.012,1:0.12,2:0.058,3:0.36,4:0.43})
        #model_final.fit_generator(train_generator,steps_per_epoch=1,epochs=epochs,verbose=1,validation_data=(val_generator),validation_steps=1,callbacks=callbacks)
        
        del model_final
        f = h5py.File(model_name, 'r+')
        del f['optimizer_weights']
        f.close()
        model_final = keras.models.load_model(model_name)
        model_to_store_path = f'{self.outdir}/{model}' 
        model_final.save(model_to_store_path)
        return model_to_store_path,train_generator.class_indices

# Hold out dataset validation function

    def inference(self,model_path,test_dir,class_dict,dim=224):
        print(test_dir)

        model = keras.models.load_model(model_path)
        test_datagen = ImageDataGenerator(preprocessing_function=pre_process)
        test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(dim,dim),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)
        filenames = test_generator.filenames
        nb_samples = len(filenames)
        pred = model.predict_generator(test_generator,steps=nb_samples)
        print(pred)
        df = pd.DataFrame()
        df['filename'] = filenames
        df['actual_class'] = df['filename'].apply(lambda x:x.split('/')[0])
        df['actual_class_index'] = df['actual_class'].apply(lambda x:int(class_dict[x]))        
        df['pred_class_index'] = np.argmax(pred,axis=1)
        k = list(class_dict.keys())
        v = list(class_dict.values())
        inv_class_dict = {}
        for k_,v_ in zip(k,v):
            inv_class_dict[v_] = k_
        df['pred_class'] =  df['pred_class_index'].apply(lambda x:(inv_class_dict[x]))
        return df

    def main(self):
        start_time = time.time()
        print('Data Processing..')
        self.num_class = len(self.class_folders)
        model_to_store_path,class_dict = self.train_model(self.train_dir,self.val_dir,n_fold=self.folds,batch_size=self.batch_size,
                                                        epochs=self.epochs,dim=self.dim,lr=self.lr,model=self.model)
        print("Model saved to dest:",model_to_store_path)

        # Validatione evaluate results
        
        folder_path = Path(f'{self.val_dir}')
        val_results_df = self.inference(model_to_store_path,folder_path,class_dict,self.dim)
        val_results_path = f'{self.outdir}/val_results.csv'
        val_results_df.to_csv(val_results_path,index=False)
        print(f'Validation results saved at : {val_results_path}') 
        pred_class_index = np.array(val_results_df['pred_class_index'].values)
        actual_class_index = np.array(val_results_df['actual_class_index'].values)
        print(pred_class_index)
        print(actual_class_index)
        accuracy = np.mean(actual_class_index == pred_class_index)
        kappa = cohen_kappa_score(pred_class_index,actual_class_index,weights='quadratic')
        #print("-----------------------------------------------------")
        print(f'Validation Accuracy: {accuracy}')
        print(f'Validation Quadratic Kappa Score: {kappa}')
        #print("-----------------------------------------------------")
        #print("Processing Time",time.time() - start_time,' secs')

if __name__ == "__main__":
    obj = TransferLearning()
    obj.main()
