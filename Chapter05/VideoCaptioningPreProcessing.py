# -*- coding: utf-8 -*-
"""
Created on Tue May 15 18:49:42 2018

@author: santanu
"""

import shutil
import subprocess
import glob
from tqdm import tqdm
import numpy as np
import cv2
import os
from keras.applications.vgg16 import VGG16
from keras.models import Model 
import fire
from elapsedtimer import ElapsedTimer

'''
VideoCaptioningPreProcessing will extract the Dense features from the 
images in the Video frame using a Pre-trained VGG16 CNN Model
'''

class VideoCaptioningPreProcessing:
    
    def __init__(self,video_dest,feat_dir,
                 temp_dest,img_dim=224,channels=3,
                 batch_size=128,frames_step=80):
        
        self.img_dim = img_dim
        self.channels = channels
        self.video_dest = video_dest
        self.feat_dir = feat_dir
        self.temp_dest = temp_dest
        self.batch_cnn = batch_size
        self.frames_step = frames_step
        
        
        

# Convert the video into image frames at a specified sampling rate 

    def video_to_frames(self,video):
        
        with open(os.devnull, "w") as ffmpeg_log:
            if os.path.exists(self.temp_dest):
                print(" cleanup: " + self.temp_dest + "/")
                shutil.rmtree(self.temp_dest)
            os.makedirs(self.temp_dest)
            video_to_frames_cmd = ["ffmpeg",
                                       
                                       '-y',
                                       '-i', video,  
                                       '-vf', "scale=400:300", 
                                       '-qscale:v', "2", 
                                       '{0}/%06d.jpg'.format(self.temp_dest)]
            subprocess.call(video_to_frames_cmd,
                            stdout=ffmpeg_log, stderr=ffmpeg_log)
                        
# Load the pre-trained VGG16 Model and extract the dense features as output 
    def model_cnn_load(self):
         model = VGG16(weights = "imagenet", include_top=True,input_shape = (self.img_dim,self.img_dim,self.channels))
         out = model.layers[-2].output
         model_final = Model(inputs=model.input,outputs=out)
         return model_final

# Load the video images 
         
    def load_image(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,(self.img_dim,self.img_dim))
        return img
        
     
# Extract the features from the pre-trained CNN    
    def extract_feats_pretrained_cnn(self):
        
        model = self.model_cnn_load()
        print('Model loaded')
            
        if not os.path.isdir(self.feat_dir):
            os.mkdir(self.feat_dir)
        #print("save video feats to %s" % (self.dir_feat))
        video_list = glob.glob(os.path.join(self.video_dest, '*.avi'))
        #print video_list 
        
        for video in tqdm(video_list):
            
            video_id = video.split("/")[-1].split(".")[0]
            print(f'Processing video {video}')
            
            #self.dest = 'cnn_feat' + '_' + video_id
            self.video_to_frames(video)
    
            image_list = sorted(glob.glob(os.path.join(self.temp_dest, '*.jpg')))
            samples = np.round(np.linspace(
                0, len(image_list) - 1,self.frames_step))
            image_list = [image_list[int(sample)] for sample in samples]
            images = np.zeros((len(image_list),self.img_dim,self.img_dim,self.channels))
            for i in range(len(image_list)):
                img = self.load_image(image_list[i])
                images[i] = img
            images = np.array(images)
            fc_feats = model.predict(images,batch_size=self.batch_cnn)
            img_feats = np.array(fc_feats)
            outfile = os.path.join(self.feat_dir, video_id + '.npy')
            np.save(outfile, img_feats)
            # cleanup
            shutil.rmtree(self.temp_dest)
            
    def process_main(self):
        self.extract_feats_pretrained_cnn()
             
            
if __name__ == '__main__':
    with ElapsedTimer('VideoCaptioningPreProcessing'):
        fire.Fire(VideoCaptioningPreProcessing)

    
