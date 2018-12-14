from __future__ import print_function, division
#import scipy
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
#import sys
#from data_loader import DataLoader
import numpy as np
import os
import time 
import glob
from scipy.misc import imread,imresize,imsave
import copy


def load_train_data(image_path, load_size=64,fine_size=64, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
            
    if not is_testing:
        img_A = imresize(img_A, [load_size, load_size])
        img_B = imresize(img_B, [load_size, load_size])
      #  h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
      #  w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
      #  img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
      #  img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = imresize(img_A, [fine_size, fine_size])
        img_B = imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1 
    img_B = img_B/127.5 - 1 

    img_AB = np.concatenate((img_A, img_B), axis=2)
        
    return img_AB

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def image_save(images, size, path):
    return imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return image_save(inverse_transform(images),size, image_path)

def inverse_transform(images):
    return (images + 1)*127.5


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image
        if np.random.rand() > 0.5:
            idx = int(np.random.rand()*self.maxsize)
            tmp1 = copy.copy(self.images[idx])[0]
            self.images[idx][0] = image[0]
            idx = int(np.random.rand()*self.maxsize)
            tmp2 = copy.copy(self.images[idx])[1]
            self.images[idx][1] = image[1]
            return [tmp1, tmp2]
        else:
	    return image


class DiscoGAN():
    
    def __init__(self):
        # Input shape
        self.lambda_l2 = 1.0
        self.image_size = 64
        self.input_dim = 3
        self.output_dim = 3
        self.batch_size = 64      
        self.df = 64
        self.gf = 64
        self.channels = 3
        self.output_c_dim = 3
        self.l_r = 2e-4
        self.beta1 = 0.5
	self.beta2 = 0.99
        self.weight_decay = 0.00001
        self.epoch = 200
        self.train_size = 10000
        self.epoch_step = 10
        self.load_size = 64
        self.fine_size = 64 
        self.checkpoint_dir = 'checkpoint_b2s'
        self.sample_dir = 'sample_b2s'
        self.print_freq = 5
        self.save_freq = 10  
        self.pool = ImagePool()
        self.model_dir = '/home/santanu/'
         
        return None
        

    def build_generator(self,image,reuse=False,name='generator'):
        
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
                
            """U-Net Generator"""
            def lrelu(x, alpha,name='lrelu'):
                with tf.variable_scope(name):
                    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
                
            def instance_norm(x,name='instance_norm'):

                with tf.variable_scope(name):
                                        
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False
                        
                    epsilon = 1e-5
                    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
                    scale = tf.get_variable('scale',[x.get_shape()[-1]], 
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
                    offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
                    out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
                    return out
                    
    
            def common_conv2d(layer_input,filters,f_size=4,stride=2,padding='SAME',norm=True,name='common_conv2d'):
                
                """Layers used during downsampling"""
                with tf.variable_scope(name):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False
                    
                    d = tf.contrib.layers.conv2d(layer_input,filters,kernel_size=f_size,stride=stride,padding=padding)
                    
                    if norm:
                        d = tf.contrib.layers.batch_norm(d)
                    d = lrelu(d,alpha=0.2)
                    return d
    
            #def common_deconv2d(layer_input,skip_input, filters,f_size=4,stride=2,dropout_rate=0,name='common_deconv2d'):
	    def common_deconv2d(layer_input,filters,f_size=4,stride=2,padding='SAME',dropout_rate=0,name='common_deconv2d'):
                """Layers used during upsampling"""
                with tf.variable_scope(name):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False


                    u = tf.contrib.layers.conv2d_transpose(layer_input,filters,f_size,stride=stride,padding=padding)
                    
                    if dropout_rate:
                        u = tf.contrib.layers.dropout(u,keep_prob=dropout_rate)
                    u = tf.contrib.layers.batch_norm(u)
                    u = tf.nn.relu(u)
                   # u = tf.contrib.keras.layers.concatenate([skip_input,u])
                    return u 
            
            
            
            # Downsampling
            dwn1 = common_conv2d(image,self.gf,stride=2,norm=False,name='dwn1')  #  64x64 -> 32x32
	    #print('dwn1',np.shape(dwn1))
            dwn2 = common_conv2d(dwn1,self.gf*2,stride=2,name='dwn2')           #  32x32 -> 16x16
	    #print('dwn2',np.shape(dwn2))
            dwn3 = common_conv2d(dwn2,self.gf*4,stride=2,name='dwn3')           #  16x16   -> 8x8
	   # print('dwn3',np.shape(dwn3))
            dwn4 = common_conv2d(dwn3,self.gf*8,stride=2,name='dwn4')           #  8x8   -> 4x4  
	   # print('dwn4',np.shape(dwn4))
            dwn5 = common_conv2d(dwn4,100,stride=1,padding='valid',name='dwn5')                 #  4x4   -> 1x1 
           # print('dwn5',np.shape(dwn5))
            
            # Upsampling
            up1 = common_deconv2d(dwn5,self.gf*8,stride=1,padding='valid',name='up1')      #  16x16    -> 16x16  
            #print(np.shape(up1))
            up2 = common_deconv2d(up1,self.gf*4,name='up2')                #  16x16    -> 32x32
            up3 = common_deconv2d(up2,self.gf*2,name='up3')               #  32x32    -> 64x64
            up4 = common_deconv2d(up3,self.gf,name='up4')                  #  64x64    -> 128x128 
			
            out_img = tf.contrib.layers.conv2d_transpose(up4,self.channels,kernel_size=4,stride=2,padding='SAME',activation_fn=tf.nn.tanh)  # 128x128 -> 256x256
           #print('out_img',(np.shape(out_img))) 
            
            return out_img

    def build_discriminator(self,image,reuse=False,name='discriminator'):
        
        
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
        
            def lrelu(x, alpha,name='lrelu'):
			
                with tf.variable_scope(name):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False

                    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
                
            def instance_norm(x,name='instance_norm'):
			
                with tf.variable_scope(name):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False

                        
                    epsilon = 1e-5
                    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
                    scale = tf.get_variable('scale',[x.get_shape()[-1]], 
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
                    offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
                    out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset
                    return out
                
    
            def d_layer(layer_input,filters,f_size=4,stride=2,norm=True,name='d_layer'):
                """Discriminator layer"""
                with tf.variable_scope(name):
                    if reuse:
                        tf.get_variable_scope().reuse_variables()
                    else:
                        assert tf.get_variable_scope().reuse is False

                    d = tf.contrib.layers.conv2d(layer_input,filters,kernel_size=f_size,stride=2, padding='SAME')
                    if norm:
                        d = tf.contrib.layers.batch_norm(d)
                    d = lrelu(d,alpha=0.2)
                    return d
    
                
            down1 = d_layer(image,self.df, norm=False,name='down1')  #256x256 -> 128x128
            #rint('down1',np.shape(down1))
            down2 = d_layer(down1,self.df*2,name='down2')         #128x128 -> 64x64
            #rint('down2',np.shape(down2))
            down3 = d_layer(down2,self.df*4,name='down3')         #64x64 -> 32x32 
            #rint('down3',np.shape(down3))
            down4 = d_layer(down3,self.df*8,name='down4')        # 32x32 -> 16x16
            #rint('down4',np.shape(down4))
    
            down5  = tf.contrib.layers.conv2d(down4,1,kernel_size=4,stride=1,padding='valid')
            #rint('down5',np.shape(down5)) 
            #rint(np.shape(down5))
            
            #logits = tf.reduce_mean(down5, [1,2,3])
            
            return down5,[down2,down3,down4]
            
    def build_network(self):
        
        def squared_loss(y_pred,labels):
            return tf.reduce_mean((y_pred - labels)**2)

        def abs_loss(y_pred,labels):
            return tf.reduce_mean(tf.abs(y_pred - labels))  


        def binary_cross_entropy_loss(logits,labels):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
        
        def feature_matching_loss(real_img_feats,fake_img_feats):
            loss = 0
            for r,f in zip(real_img_feats,fake_img_feats):
                loss += (tf.reduce_mean(r) - tf.reduce_mean(f))**2
            return loss 
            

       
        self.images_real = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,self.input_dim + self.output_dim])
        
        self.image_real_A = self.images_real[:,:,:,:self.input_dim]
        self.image_real_B = self.images_real[:,:,:,self.input_dim:self.input_dim + self.output_dim]
        self.images_fake_B = self.build_generator(self.image_real_A,reuse=False,name='generator_AB')
        self.images_fake_A = self.build_generator(self.images_fake_B,reuse=False,name='generator_BA')
        self.images_fake_A_ = self.build_generator(self.image_real_B,reuse=True,name='generator_BA')
        self.images_fake_B_ = self.build_generator(self.images_fake_A_,reuse=True,name='generator_AB')
        
        self.D_B_fake,self.D_B_fake_feat = self.build_discriminator(self.images_fake_B ,reuse=False, name="discriminatorB")
        self.D_A_fake,self.D_A_fake_feat =  self.build_discriminator(self.images_fake_A_,reuse=False, name="discriminatorA") 

        self.D_B_real,self.D_B_real_feat  = self.build_discriminator(self.image_real_B,reuse=True, name="discriminatorB")
        self.D_A_real,self.D_A_real_feat  = self.build_discriminator(self.image_real_A,reuse=True, name="discriminatorA")
        
        
         
        
        self.loss_GABA = self.lambda_l2*squared_loss(self.images_fake_A,self.image_real_A) + (0.1*binary_cross_entropy_loss(labels=tf.ones_like(self.D_B_fake),logits=self.D_B_fake) + 
                                                                                               0.9*feature_matching_loss(self.D_B_real_feat,self.D_B_fake_feat))
        self.loss_GBAB = self.lambda_l2*squared_loss(self.images_fake_B_,self.image_real_B) + (0.1*binary_cross_entropy_loss(labels=tf.ones_like(self.D_A_fake),logits=self.D_A_fake) +
                                                                                               0.9*feature_matching_loss(self.D_A_real_feat,self.D_A_fake_feat)) 
        self.generator_loss = self.loss_GABA + self.loss_GBAB
              
                
        self.D_B_loss_real = binary_cross_entropy_loss(tf.ones_like(self.D_B_real),self.D_B_real)
        self.D_B_loss_fake = binary_cross_entropy_loss(tf.zeros_like(self.D_B_fake),self.D_B_fake)
        self.D_B_loss = (self.D_B_loss_real + self.D_B_loss_fake) / 2.0
        
        
        self.D_A_loss_real = binary_cross_entropy_loss(tf.ones_like(self.D_A_real),self.D_A_real)
        self.D_A_loss_fake = binary_cross_entropy_loss(tf.zeros_like(self.D_A_fake),self.D_A_fake)
        self.D_A_loss = (self.D_A_loss_real + self.D_A_loss_fake) / 2.0
        
        self.discriminator_loss = self.D_B_loss + self.D_A_loss
        
        self.loss_GABA_sum = tf.summary.scalar("g_loss_a2b", self.loss_GABA)
        self.loss_GBAB_sum = tf.summary.scalar("g_loss_b2a", self.loss_GBAB)
        self.g_total_loss_sum = tf.summary.scalar("g_loss", self.generator_loss)
        self.g_sum = tf.summary.merge([self.loss_GABA_sum,self.loss_GBAB_sum,self.g_total_loss_sum])
        
        self.loss_db_sum = tf.summary.scalar("db_loss", self.D_B_loss)
        self.loss_da_sum = tf.summary.scalar("da_loss", self.D_A_loss)
        self.loss_d_sum = tf.summary.scalar("d_loss",self.discriminator_loss)
        
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.D_B_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.D_B_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.D_A_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.D_A_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.loss_da_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.loss_db_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.loss_d_sum]
        )

        
        trainable_variables = tf.trainable_variables()
        
        self.d_variables = [var for var in trainable_variables if 'discriminator' in var.name]
        self.g_variables = [var for var in trainable_variables if 'generator' in var.name]
      
        print ('Variable printing start :'  )
        for var in self.d_variables: 
            print(var.name)
            
        self.test_image_A = tf.placeholder(tf.float32,[None, self.image_size,self.image_size,self.input_dim], name='test_A')
        self.test_image_B = tf.placeholder(tf.float32,[None, self.image_size, self.image_size,self.output_c_dim], name='test_B')
        self.saver = tf.train.Saver()
        
        
    def train_network(self):
        
        self.learning_rate = tf.placeholder(tf.float32)
        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1,beta2=self.beta2).minimize(self.discriminator_loss,var_list=self.d_variables)
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate,beta1=self.beta1,beta2=self.beta2).minimize(self.generator_loss,var_list=self.g_variables)      
        
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init_op)
        self.dataset_dir_bags = '/home/santanu/Downloads/DiscoGAN/edges2handbags/train/trainB/'
        self.dataset_dir_shoes = '/home/santanu/Downloads/DiscoGAN/edges2shoes/trainB/'
        self.writer = tf.summary.FileWriter("./b2s_logs", self.sess.graph)
        count = 1
        start_time = time.time()
        
        for epoch in range(self.epoch):
            data_A = os.listdir(self.dataset_dir_bags)
            data_B = os.listdir(self.dataset_dir_shoes)
            data_A = [ (self.dataset_dir_bags + str(file_name)) for file_name in data_A ] 

            data_B = [ (self.dataset_dir_shoes + str(file_name)) for file_name in data_B ]            
            np.random.shuffle(data_A)
            np.random.shuffle(data_B)
            batch_ids = min(min(len(data_A), len(data_B)), self.train_size) // self.batch_size
#            lr = self.l_r if epoch < self.epoch_step else self.l_r*(self.epoch-epoch)/(self.epoch-self.epoch_step)
            lr = self.l_r if epoch < self.epoch_step else self.l_r*(self.epoch-epoch)/(self.epoch-self.epoch_step)
            
            for id_ in range(0, batch_ids):
                batch_files = list(zip(data_A[id_ * self.batch_size:(id_ + 1) * self.batch_size],
                                      data_B[id_ * self.batch_size:(id_ + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, self.load_size, self.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)
    
                    # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                        [self.images_fake_A_,self.images_fake_B,self.g_optimizer,self.g_sum],
                        feed_dict={self.images_real: batch_images, self.learning_rate:lr})
                self.writer.add_summary(summary_str, count)
                [fake_A,fake_B] = self.pool([fake_A, fake_B])
    
                    # Update D network
                _, summary_str = self.sess.run(
                        [self.d_optimizer,self.d_sum],
                        feed_dict={self.images_real: batch_images,
                               #    self.fake_A_sample: fake_A,
                               #    self.fake_B_sample: fake_B,
                                   self.learning_rate: lr})
                self.writer.add_summary(summary_str, count)
    
                count += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                        epoch, id_, batch_ids, time.time() - start_time)))
    
                if count % self.print_freq == 1:
                    self.sample_model(self.sample_dir, epoch, id_)
    
                if count % self.save_freq == 2:
                    self.save_model(self.checkpoint_dir, count)

    
      
    def save_model(self,checkpoint_dir,step):
        model_name = "discogan_b2s.model"
        model_dir = "%s_%s" % (self.model_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
        self.saver.save(self.sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

    def load_model(self,checkpoint_dir):
        
        print(" [*] Reading checkpoint...")
    
        model_dir = "%s_%s" % (self.model_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
     
   
        
    def sample_model(self, sample_dir, epoch, id_):
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

           
        data_A = os.listdir(self.dataset_dir_bags)
        data_B = os.listdir(self.dataset_dir_shoes) 
        data_A = [ (self.dataset_dir_bags + str(file_name)) for file_name in data_A ]
        data_B = [ (self.dataset_dir_shoes + str(file_name)) for file_name in data_B ]
 

        np.random.shuffle(data_A)
        np.random.shuffle(data_B)
        batch_files = list(zip(data_A[:self.batch_size], data_B[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.images_fake_A_,self.images_fake_B],
            feed_dict={self.images_real: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, id_))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, id_))


if __name__ == '__main__':
    gan = DiscoGAN()
    gan.build_network()
   
    gan.train_network()
    
    
    
