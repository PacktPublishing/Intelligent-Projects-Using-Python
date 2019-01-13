from claptcha import Claptcha
import pandas as pd
import os
import numpy as np
import cv2
import keras
from keras.models import Sequential,Model
from keras.layers import Dropout,Activation, \
        Convolution2D, GlobalAveragePooling2D, merge,MaxPooling2D,Conv2D,Flatten,Dense,Input
from keras import backend as K 
from keras.optimizers import Adam
import fire
from elapsedtimer import ElapsedTimer
from keras.utils import plot_model

def create_dict_char_to_index():
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'.upper()
    chars = list(chars)
    index = np.arange(len(chars))
    char_to_index_dict,index_to_char_dict = {},{}
    for v,k in zip(index,chars):
        char_to_index_dict[k] = v
        index_to_char_dict[v] = k

    return char_to_index_dict,index_to_char_dict
    
    
    
def load_img(path,dim=(100,40)):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,dim)
    img = img.reshape((dim[1],dim[0],1))
    #print(img.shape)
    return img/255.


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dest,char_to_index_dict,batch_size=32,n_classes=36,dim=(40,100,1),shuffle=True):
        'Initialization'
        self.dest = dest
        self.files = os.listdir(self.dest)
        self.char_to_index_dict = char_to_index_dict
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

        # Generate data
        X, y = self.__data_generation(list_files)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self,list_files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        dim_h = self.dim[0]
        dim_w = self.dim[1]//4
        channels = self.dim[2]
        X = np.empty((4*len(list_files),dim_h,dim_w,channels))
        y = np.empty((4*len(list_files)),dtype=int)
#        print(X.shape,y.shape)

        # Generate data
        k = -1 
        for f in list_files:
            target = list(f.split('.')[0])
            target = [self.char_to_index_dict[c] for c in target]
            img = load_img(self.dest + f)
            img_h,img_w = img.shape[0],img.shape[1]
            crop_w = img.shape[1]//4
            for i in range(4):
                img_crop = img[:,i*crop_w:(i+1)*crop_w]
                k+=1
                X[k,] = img_crop
                y[k] = int(target[i])
			
        return X,y
           
        
    
def _model_(n_classes):
    # Build the neural network
    input_ = Input(shape=(40,25,1)) 
    
    # First convolutional layer with max pooling
    x = Conv2D(20, (5, 5), padding="same",activation="relu")(input_)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    # Second convolutional layer with max pooling
    x = Conv2D(50, (5, 5), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.2)(x)
    # Hidden layer with 1024 nodes
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    # Output layer with 36 nodes (one for each possible alphabet/digit we predict)
    out = Dense(n_classes,activation='softmax')(x)
    model = Model(inputs=[input_],outputs=out)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    plot_model(model,show_shapes=True, to_file='model.png')
    return model 


def train(dest_train,dest_val,outdir,batch_size,n_classes,dim,shuffle,epochs,lr):
    char_to_index_dict,index_to_char_dict = create_dict_char_to_index()
    model = _model_(n_classes)
    from keras.utils import plot_model
    plot_model(model, to_file=outdir + 'model.png')
    train_generator =  DataGenerator(dest_train,char_to_index_dict,batch_size,n_classes,dim,shuffle)
    val_generator =  DataGenerator(dest_val,char_to_index_dict,batch_size,n_classes,dim,shuffle)
    model.fit_generator(train_generator,epochs=epochs,validation_data=val_generator)
    model.save(outdir + 'captcha_breaker.h5')

def evaluate(model_path,eval_dest,outdir,fetch_target=True):
    char_to_index_dict,index_to_char_dict = create_dict_char_to_index()
    files = os.listdir(eval_dest)
    model = keras.models.load_model(model_path)
    predictions,targets = [],[]
    
    for f in files:
        if fetch_target == True:
            target = list(f.split('.')[0])
            targets.append(target)

        pred = []
        img = load_img(eval_dest + f)
        img_h,img_w = img.shape[0],img.shape[1]
        crop_w = img.shape[1]//4
        for i in range(4):
            img_crop = img[:,i*crop_w:(i+1)*crop_w]
            img_crop = img_crop[np.newaxis,:]
            pred_index  = np.argmax(model.predict(img_crop),axis=1)
            #print(pred_index)
            pred_char   = index_to_char_dict[pred_index[0]]
            pred.append(pred_char)
        predictions.append(pred)

    df = pd.DataFrame()
    df['files'] = files
    df['predictions'] = predictions

    if fetch_target == True:
        match = []
        
        df['targets'] = targets

        accuracy_count = 0
        for i in range(len(files)):
            if targets[i] == predictions[i]:
                accuracy_count+= 1
                match.append(1)
            else:
                match.append(0)
        print(f'Accuracy: {accuracy_count/float(len(files))} ')
        
        eval_file = outdir + 'evaluation.csv'
        df['match'] = match
        df.to_csv(eval_file,index=False)
        print(f'Evaluation file written at: {eval_file} ')
       

 
if __name__ == '__main__':
    with ElapsedTimer('captcha_solver'):
        fire.Fire()
