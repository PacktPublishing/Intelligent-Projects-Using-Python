# Deep Convolutional GAN with Keras 
# GANs Components -> Generator G  Converts noise z to fake images G(z) similar to real images x 
#                    Discriminator D -> Tries to distinguish between real (x) and fake images generated(G(z)) 
#                    Learning Mechanism -> Both the Generator and Discriminator learns by playing a zero sum min-max game.
#                    Solution is the Saddle point solution of the classification logloss of Real and Fake images.
#                    The saddle point would be a minima point with respect to the Discriminator and maxima with respect to the Generator
#                    This saddle point in terms of Game theory is the famous Nash Equilibrium. At the Nash equilibrium the probability distribution 
#                    of the Generator Images should match the probability distribution of the Real Images i.e.
#                    P(G(z)) ~  P(x)
           

# Load the libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.core import Flatten
from keras.optimizers import SGD,Adam
from scipy.io import loadmat
import numpy as np
from PIL import Image
import argparse
import math
import cv2
import os
import fire
from elapsedtimer import ElapsedTimer 
import matplotlib.pyplot as plt
from keras.utils import plot_model


def load_img(path,dim=(32,32)):

    img = cv2.imread(path)
    img = cv2.resize(img,dim)
    img = img.reshape((dim[1],dim[0],3))
    return img

# Define the Generator Network

def generator(input_dim,alpha=0.2):
    model = Sequential()
    model.add(Dense(input_dim=input_dim, output_dim=4*4*512))
    model.add(Reshape(target_shape=(4,4,512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Conv2DTranspose(256, kernel_size=5, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Conv2DTranspose(128, kernel_size=5, strides=2, padding='same'))   
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Conv2DTranspose(3, kernel_size=5, strides=2, padding='same'))   
    model.add(Activation('tanh'))
    return model

#Define the Discriminator Network


def discriminator(img_dim,alpha=0.2):
    model = Sequential()
    model.add(
            Conv2D(64, kernel_size=5,strides=2,
            padding='same',
            input_shape=img_dim)
            )
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(128,kernel_size=5,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Conv2D(256,kernel_size=5,strides=2,padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# Define a combination of Generator and Discriminator 

def generator_discriminator(g, d):
    model = Sequential()
    model.add(g)
#    d.trainable = False
    model.add(d)
    return model

# Steps to output the quality of images generated at intervals

def combine_images(generated_images,outdir,epoch,index):
    n_images = len(generated_images)
    n_images = min(30,n_images)
    cols = 10
    rows = n_images//cols
    
    plt.figure(figsize=(10, 10))
    for i in range(n_images):
        img = generated_images[i]
        img = np.uint8(((img+1)/2)*255)
        ax = plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(outdir + str(epoch)+"_"+str(index)+".png")

def read_data(dest,dir_flag=False):
    if dir_flag == True:
        
        files = os.listdir(dest)
        X = []
        
        for f in files:
            img = load_img(dest + f)
            X.append(img)
    else:
        train_data = loadmat(dest)
        X,y = train_data['X'], train_data['y']
        X = np.rollaxis(X,3) 
        X = (X/255)*2-1
    return X



def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable

# Training the Discriminator and Generator 

def train(dest_train,dir_flag:bool,outdir,
        gen_input_dim,gen_lr,gen_beta1,
        dis_input_dim,dis_lr,dis_beta1,
        epochs,batch_size,alpha=0.2,smooth_coef=0.1):

    X_train = read_data(dest_train,dir_flag)
    #Image pixels are normalized between -1 to +1 so that one can use the tanh activation function
    #_train = (X_train.astype(np.float32) - 127.5)/127.5
    g = generator(gen_input_dim,alpha)
    plot_model(g,show_shapes=True, to_file='generator_model.png')
    d = discriminator(dis_input_dim,alpha)
    d_optim = Adam(lr=dis_lr,beta_1=dis_beta1)
    d.compile(loss='binary_crossentropy',optimizer=d_optim)
    plot_model(d,show_shapes=True, to_file='discriminator_model.png')
    g_d = generator_discriminator(g, d)
    g_optim = Adam(lr=gen_lr,beta_1=gen_beta1)
    g_d.compile(loss='binary_crossentropy', optimizer=g_optim)
    plot_model(g_d,show_shapes=True, to_file='generator_discriminator_model.png')
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/batch_size))
        for index in range(int(X_train.shape[0]/batch_size)):
            noise = np.random.normal(loc=0, scale=1, size=(batch_size,gen_input_dim))
            image_batch = X_train[index*batch_size:(index+1)*batch_size,:]
            generated_images = g.predict(noise, verbose=0)
            if index % 20 == 0:
                combine_images(generated_images,outdir,epoch,index)
                # Images converted back to be within 0 to 255  
                #mage = image*127.5+127.5
            print(image_batch.shape,generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            d1 = d.train_on_batch(image_batch,[1 - smooth_coef]*batch_size)
            d2 = d.train_on_batch(generated_images,[0]*batch_size)

            y = [1] * batch_size + [0] * batch_size
            # Train the Discriminator on both real and fake images 
            make_trainable(d,True)
            #_loss = d.train_on_batch(X, y)
            d_loss = d1 + d2

            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.normal(loc=0, scale=1, size=(batch_size,gen_input_dim))
            make_trainable(d,False)
            #d.trainable = False
            # Train the generator on fake images from Noise   
            g_loss = g_d.train_on_batch(noise, [1] * batch_size)
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)

# Generate Captchas for use

def generate_captcha(gen_input_dim,alpha,
             num_images,model_dir,outdir):

    g = generator(gen_input_dim,alpha)
    g.load_weights(model_dir + 'generator')
    noise = np.random.normal(loc=0, scale=1, size=(num_images,gen_input_dim))
    generated_images = g.predict(noise, verbose=1)
    for i in range(num_images):
        img = generated_images[i,:]
        img = np.uint8(((img+1)/2)*255)
        img = Image.fromarray(img)
        img.save(outdir + 'captcha_' + str(i) + '.png') 
    

if __name__ == '__main__':
    with ElapsedTimer('main'):
        fire.Fire()
