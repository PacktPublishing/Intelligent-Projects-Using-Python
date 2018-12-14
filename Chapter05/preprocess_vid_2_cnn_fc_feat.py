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

class video_captioning:
    
    def __init__(self):
        
        self.img_dim = 224
        self.channels = 3
        self.video_dest = '/home/santanu/Downloads/Video Captioning/data/'
        self.dir_feat = '/home/santanu/Downloads/Video Captioning/rgb_feats/'
        self.dest = '/home/santanu/Downloads/Video Captioning/dest_temp/'
        self.batch_cnn = 128
        self.frames_step = 80
        
        return None
        
        
        


# Convert the video into image frames at a specified sampling rate 

    def video_to_frames(self,video):
        
        with open(os.devnull, "w") as ffmpeg_log:
            if os.path.exists(self.dest):
                print(" cleanup: " + self.dest + "/")
                shutil.rmtree(self.dest)
            os.makedirs(self.dest)
            video_to_frames_cmd = ["ffmpeg",
                                       
                                       '-y',
                                       '-i', video,  
                                       '-vf', "scale=400:300", 
                                       '-qscale:v', "2", 
                                       '{0}/%06d.jpg'.format(self.dest)]
            subprocess.call(video_to_frames_cmd,
                            stdout=ffmpeg_log, stderr=ffmpeg_log)
                        

    def model_cnn_load(self):
         model = VGG16(weights = "imagenet", include_top=True, input_shape = (self.img_dim,self.img_dim,self.channels))
         out = model.layers[-2].output
         model_final = Model(input=model.input,output=out)
         return model_final
         
    def load_image(self,path):
        img = cv2.imread(path)
        img = cv2.resize(img,(self.img_dim,self.img_dim))
        return img
        
     
                        

    def extract_feats_pretrained_cnn(self):
        
        model = self.model_cnn_load()
        print 'Model loaded'
            
        if not os.path.isdir(self.dir_feat):
            os.mkdir(self.dir_feat)
        #print("save video feats to %s" % (self.dir_feat))
        video_list = glob.glob(os.path.join(self.video_dest, '*.avi'))
        #print video_list 
        
        for video in tqdm(video_list):
            
            video_id = video.split("/")[-1].split(".")[0]
            print video
            
            #self.dest = 'cnn_feat' + '_' + video_id
            self.video_to_frames(video)
    
            image_list = sorted(glob.glob(os.path.join(self.dest, '*.jpg')))
            samples = np.round(np.linspace(
                0, len(image_list) - 1,self.frames_step))
            image_list = [image_list[int(sample)] for sample in samples]
            images = np.zeros((len(image_list),self.img_dim,self.img_dim,self.channels))
            for i in range(len(image_list)):
                img = self.load_image(image_list[i])
                images[i] = img
            images = np.array(images)
            print np.shape(images)
            fc_feats = model.predict(images,batch_size=self.batch_cnn)
            img_feats = np.array(fc_feats)
            outfile = os.path.join(self.dir_feat, video_id + '.npy')
            np.save(outfile, img_feats)
            # cleanup
            shutil.rmtree(self.dest)
            
    def process_main(self):
        self.extract_feats_pretrained_cnn()
             
            
if __name__ == '__main__':
    vc = video_captioning()
    vc.process_main()
    
