import numpy as np
import pandas as pd
import tensorflow as tf
import sys
import os
print tf.__version__

class recommender:
    
    def __init__(self,infile):
           
        self.train_file = '/home/santanu/Downloads/RBM Recommender/ml-100k/train_data.npy'
        self.data = np.load(infile)

        if sys.argv[1] == 'train':
           self.train_file = infile
           self.data = np.load(infile)
        else:
           #elf.test_file = infile
           self.data = np.load(infile)
           self.user_index = list(self.data[:,0]) 
           self.movie_index = list(self.data[:,1])
           self.rating_index = list(self.data[:,2])   
           self.train_data = np.load(self.train_file)
           self.test_data =  self.train_data[self.user_index,:,:]
            
             

        #self.data = np.load(infile) 
        self.ranks = 5
        self.batch_size = 32
        self.epochs = 500
        self.learning_rate = 1e-4
        self.users = self.train_data.shape[0]
        self.num_hidden = 500
        self.num_movies = self.train_data.shape[1]
        self.num_ranks = 5
	self.display_step = 1
        self.path_save = sys.argv[3]
    
    def next_batch(self):
        while True:
            ix = np.random.choice(np.arange(self.data.shape[0]),self.batch_size)
            train_X  = self.data[ix,:,:]   
            yield train_X
        
        
    def __network(self):
        
        self.x  = tf.placeholder(tf.float32, [None,self.num_movies,self.num_ranks], name="x") 
        self.xr = tf.reshape(self.x, [-1,self.num_movies*self.num_ranks], name="xr") 
        self.W  = tf.Variable(tf.random_normal([self.num_movies*self.num_ranks,self.num_hidden], 0.01), name="W") 
        self.b_h = tf.Variable(tf.zeros([1,self.num_hidden],  tf.float32, name="b_h")) 
        self.b_v = tf.Variable(tf.zeros([1,self.num_movies*self.num_ranks],tf.float32, name="b_v")) 
        self.k = 2

## Converts the probability into discrete binary states i.e. 0 and 1 
        def sample_hidden(probs):
            return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

        def sample_visible(logits):
        
            logits = tf.reshape(logits,[-1,self.num_ranks])
            sampled_logits = tf.multinomial(logits,1)             
            sampled_logits = tf.one_hot(sampled_logits,depth = 5)
            logits = tf.reshape(logits,[-1,self.num_movies*self.num_ranks])
            print logits
            return logits  
    
                      

          
  
## Gibbs sampling step
        def gibbs_step(x_k):
          #  x_k = tf.reshape(x_k,[-1,self.num_movies*self.num_ranks]) 
            h_k = sample_hidden(tf.sigmoid(tf.matmul(x_k,self.W) + self.b_h))
            x_k = sample_visible(tf.add(tf.matmul(h_k,tf.transpose(self.W)),self.b_v))
            return x_k
## Run multiple gives Sampling step starting from an initital point     
        def gibbs_sample(k,x_k):
             
            for i in range(k):
                x_k = gibbs_step(x_k) 
# Returns the gibbs sample after k iterations
            return x_k

# Constrastive Divergence algorithm
# 1. Through Gibbs sampling locate a new visible state x_sample based on the current visible state x    
# 2. Based on the new x sample a new h as h_sample    
        self.x_s = gibbs_sample(self.k,self.xr) 
        self.h_s = sample_hidden(tf.sigmoid(tf.matmul(self.x_s,self.W) + self.b_h)) 

# Sample hidden states based given visible states
        self.h = sample_hidden(tf.sigmoid(tf.matmul(self.xr,self.W) + self.b_h)) 
# Sample visible states based given hidden states
        self.x_ = sample_visible(tf.matmul(self.h,tf.transpose(self.W)) + self.b_v)

# The weight updated based on gradient descent 
        #self.size_batch = tf.cast(tf.shape(x)[0], tf.float32)
        self.W_add  = tf.multiply(self.learning_rate/self.batch_size,tf.subtract(tf.matmul(tf.transpose(self.xr),self.h),tf.matmul(tf.transpose(self.x_s),self.h_s)))
        self.bv_add = tf.multiply(self.learning_rate/self.batch_size, tf.reduce_sum(tf.subtract(self.xr,self.x_s), 0, True))
        self.bh_add = tf.multiply(self.learning_rate/self.batch_size, tf.reduce_sum(tf.subtract(self.h,self.h_s), 0, True))
        self.updt = [self.W.assign_add(self.W_add), self.b_v.assign_add(self.bv_add), self.b_h.assign_add(self.bh_add)]
        
        
    def _train(self):
            
        self.__network()
# TensorFlow graph execution

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=100,write_version=1)
            #saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)  
            # Initialize the variables of the Model
            init = tf.global_variables_initializer()
            sess.run(init)
            
            total_batches = self.data.shape[0]//self.batch_size
            batch_gen = self.next_batch()
            # Start the training 
            for epoch in range(self.epochs):
                if epoch < 150:
                    k = 2
    
                if (epoch > 150) & (epoch < 250):
                    k = 3
                    
                if (epoch > 250) & (epoch < 350):
                    k = 5
    
                if (epoch > 350) & (epoch < 500):
                    k = 9
                
                    # Loop over all batches
                for i in range(total_batches):
                    self.X_train = next(batch_gen)
                    # Run the weight update 
                    #batch_xs = (batch_xs > 0)*1
                    _ = sess.run([self.updt],feed_dict={self.x:self.X_train})
                    
                # Display the running step 
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1))
                    saver.save(sess, os.path.join(self.path_save,'model'), global_step=epoch)    
                          
	print("RBM training Completed !") 



    def _inference(self):
        
	self.model_path = sys.argv[3]
               
	#self.test_data = self.data
        self.__network()
        sess = tf.Session()

        saver = tf.train.Saver(tf.all_variables(), reshape=True)
	saver.restore(sess,self.model_path)  
        x_ = tf.matmul(self.h,tf.transpose(self.W)) + self.b_v
        #print x_
        logits = tf.reshape(x_,[-1,self.num_ranks])   
       # print logits
        logits = tf.argmax(logits,axis=-1)
       # print logits
        logits = tf.reshape(logits,[-1,self.num_movies])
        out = sess.run(logits,feed_dict={self.x:self.test_data})
        ratings_pred = []
        i = 0  
        for x in self.movie_index:
            pred = out[i,x] + 1
            ratings_pred.append(pred) 
            i+=1 
         
        ratings_pred = np.array(ratings_pred) 
        ratings_pred = np.reshape(ratings_pred,(-1,1)) 
        print ratings_pred.shape
        print self.data.shape  
        out = np.hstack((self.data,ratings_pred))
        out = pd.DataFrame(out)
        print out 
        out.columns=['User','Movie','Actual Rating','Predicted Rating']
        return out 
 
        
              


    
if __name__ == '__main__':

    if sys.argv[1] == 'train':
  
        infile = sys.argv[2]
        model = recommender(infile)
        model._train()

    if sys.argv[1] == 'test':
        
        infile = sys.argv[2]
        
        model = recommender(infile) 
        out = model._inference()
        out.to_csv('/home/santanu/Downloads/RBM Recommender/results.csv') 

       
         

	
