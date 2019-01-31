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
import h5py
import argparse
from sklearn.externals import joblib
import json
import keras 
from pathlib import Path

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
	   

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,files,labels,batch_size=32,n_classes=5,dim=(224,224,3),shuffle=True):
        'Initialization'
        self.labels = labels
        self.files = files
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of files to be processed in the batch
        list_files = [self.files[k] for k in indexes]
        labels    = [self.labels[k] for k in indexes] 

        # Generate data
        X, y = self.__data_generation(list_files,labels)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,list_files,labels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((len(list_files),self.dim[0],self.dim[1],self.dim[2]))
        y = np.empty((len(list_files)),dtype=int)
     #   print(X.shape,y.shape)

        # Generate data
        k = -1 
        for i,f in enumerate(list_files):
            #      print(f)
            img = get_im_cv2(f,dim=self.dim[0])
            img = pre_process(img)
            label = labels[i]
            #label = keras.utils.np_utils.to_categorical(label,self.n_classes)
            X[i,] = img
            y[i,] = label
       # print(X.shape,y.shape)    
        return X,y



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
		parser.add_argument('--mode',help='train or validation',default='train')
		parser.add_argument('--model_save_dest',help='dict wit model paths')
		parser.add_argument('--outdir',help='output directory')
		
		args = parser.parse_args()
		self.path = args.path
		self.class_folders = json.loads(args.class_folders)
		self.dim  = int(args.dim)
		self.lr   = float(args.lr)
		self.batch_size = int(args.batch_size)
		self.epochs =  int(args.epochs)
		self.initial_layers_to_freeze = int(args.initial_layers_to_freeze)
		self.model = args.model
		self.folds = int(args.folds)
		self.mode = args.mode
		self.model_save_dest = args.model_save_dest
		self.outdir = args.outdir
	
	
	def get_im_cv2(self,path,dim=224):
		img = cv2.imread(path)
		resized = cv2.resize(img, (dim,dim), cv2.INTER_LINEAR)
		return resized

	# Pre Process the Images based on the ImageNet pre-trained model Image transformation
	def pre_process(self,img):
		img[:,:,0] = img[:,:,0] - 103.939
		img[:,:,1] = img[:,:,0] - 116.779
		img[:,:,2] = img[:,:,0] - 123.68
		return img
	   
	# Function to build X, y in numpy format based on the train/validation datasets
	def read_data(self,class_folders,path,num_class,dim,train_val='train'):
		labels = []
		file_list = []
		for c in class_folders:
			path_class = path + str(train_val) + '/' + str(c)
			files = os.listdir(path_class)
			files = [(path_class + '/' + f) for f in files]
			file_list += files
			labels += len(files)*[int(c.split('class')[1])]

		return file_list,labels
		
	def inception_pseudo(self,dim=224,freeze_layers=30,full_freeze='N'):
		model = InceptionV3(weights='imagenet',include_top=False)
		x = model.output
		x = GlobalAveragePooling2D()(x)
		x = Dense(512, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(512, activation='relu')(x)
		x = Dropout(0.5)(x)
		out = Dense(1)(x)
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
		out = Dense(1)(x)
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
		out = Dense(1)(x)
		model_final = Model(input = model.input,outputs=out)
		if full_freeze != 'N':
			for layer in model.layers[0:freeze_layers]:
				layer.trainable = False
		return model_final


	def train_model(self,file_list,labels,n_fold=5,batch_size=16,epochs=40,dim=224,lr=1e-5,model='ResNet50'):
		model_save_dest = {}
		k = 0
		kf = KFold(n_splits=n_fold, random_state=0, shuffle=True)

		for train_index,test_index in kf.split(file_list):


			k += 1
			file_list = np.array(file_list)
			labels   = np.array(labels)
			train_files,train_labels  = file_list[train_index],labels[train_index]
			val_files,val_labels  = file_list[test_index],labels[test_index]
			
			if model == 'Resnet50':
				model_final = self.resnet_pseudo(dim=224,freeze_layers=10,full_freeze='N')
			
			if model == 'VGG16':
				model_final = self.VGG16_pseudo(dim=224,freeze_layers=10,full_freeze='N') 
			
			if model == 'InceptionV3':
				model_final = self.inception_pseudo(dim=224,freeze_layers=10,full_freeze='N')
				
			adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
			model_final.compile(optimizer=adam, loss=["mse"],metrics=['mse'])
			reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50,patience=3, min_lr=0.000001)
			early = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
			logger = CSVLogger('keras-5fold-run-01-v1-epochs_ib.log', separator=',', append=False)
			checkpoint = ModelCheckpoint(
								'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check',
								monitor='val_loss', mode='min',
								save_best_only=True,
								verbose=1) 
			callbacks = [reduce_lr,early,checkpoint,logger]
			train_gen = DataGenerator(train_files,train_labels,batch_size=32,n_classes=len(self.class_folders),dim=(self.dim,self.dim,3),shuffle=True)
			val_gen = DataGenerator(val_files,val_labels,batch_size=32,n_classes=len(self.class_folders),dim=(self.dim,self.dim,3),shuffle=True)
			model_final.fit_generator(train_gen,epochs=epochs,verbose=1,validation_data=(val_gen),callbacks=callbacks)
			model_name = 'kera1-5fold-run-01-v1-fold-' + str('%02d' % (k + 1)) + '-run-' + str('%02d' % (1 + 1)) + '.check'
			del model_final
			f = h5py.File(model_name, 'r+')
			del f['optimizer_weights']
			f.close()
			model_final = keras.models.load_model(model_name)
			model_name1 = self.outdir + str(model) + '___' + str(k) 
			model_final.save(model_name1)
			model_save_dest[k] = model_name1
				
		return model_save_dest

	# Hold out dataset validation function

	def inference_validation(self,test_X,test_y,model_save_dest,n_class=5,folds=5):
		print(test_X.shape,test_y.shape)
		pred = np.zeros(test_X.shape[0])
		for k in range(1,folds + 1):
			print(f'running inference on fold: {k}')
			model = keras.models.load_model(model_save_dest[k])
			pred = pred + model.predict(test_X)[:,0]
			pred = pred
			print(pred.shape)
			print(pred)
		pred = pred/float(folds)
		pred_class = np.round(pred)
		pred_class = np.array(pred_class,dtype=int)
		pred_class = list(map(lambda x:4 if x > 4 else x,pred_class))
		pred_class = list(map(lambda x:0 if x < 0 else x,pred_class))
		act_class = test_y 
		accuracy = np.sum([pred_class == act_class])*1.0/len(test_X)
		kappa = cohen_kappa_score(pred_class,act_class,weights='quadratic')
		return pred_class,accuracy,kappa   
	
	def main(self):
		start_time = time.time()
		self.num_class = len(self.class_folders)
		if self.mode == 'train':
			print("Data Processing..")
			file_list,labels= self.read_data(self.class_folders,self.path,self.num_class,self.dim,train_val='train')
			print(len(file_list),len(labels))
			print(labels[0],labels[-1])
			self.model_save_dest = self.train_model(file_list,labels,n_fold=self.folds,batch_size=self.batch_size,
                                                        epochs=self.epochs,dim=self.dim,lr=self.lr,model=self.model)
			joblib.dump(self.model_save_dest,f'{self.outdir}/model_dict.pkl')
			print("Model saved to dest:",self.model_save_dest)
		else:
			model_save_dest = joblib.load(self.model_save_dest)
			print('Models loaded from:',model_save_dest)
            # Do inference/validation
			test_files,test_y = self.read_data(self.class_folders,self.path,self.num_class,self.dim,train_val='validation')
			test_X = []
			for f in test_files:
				img = self.get_im_cv2(f)
				img = self.pre_process(img)
				test_X.append(img)
			test_X = np.array(test_X)
			test_y = np.array(test_y)
			print(test_X.shape)
			print(len(test_y))
			pred_class,accuracy,kappa = self.inference_validation(test_X,test_y,model_save_dest,n_class=self.num_class,folds=self.folds)
			results_df = pd.DataFrame()
			results_df['file_name'] = test_files
			results_df['target'] = test_y
			results_df['prediction'] = pred_class
			results_df.to_csv(f'{self.outdir}/val_resuts_reg.csv',index=False)
			
			print("-----------------------------------------------------")
			print("Kappa score:", kappa)
			print("accuracy:", accuracy) 
			print("End of training")
			print("-----------------------------------------------------")
			print("Processing Time",time.time() - start_time,' secs')
		
if __name__ == "__main__":
	obj = TransferLearning()
	obj.main()


		
		
