# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 01:53:44 2018

@author: santanu
"""

import tensorflow as tf
import numpy as np
import os 
import keras 
import pickle
import fire
from elapsedtimer import ElapsedTimer

'''
Train Model for Movie Review
'''

class Review_sentiment:
    
    def __init__(self,path,epochs):
        self.batch_size = 250
        self.train_to_val_ratio = 5.0
        self.batch_size_val =  int(self.batch_size/self.train_to_val_ratio)
        self.epochs = epochs
        self.hidden_states = 100
        self.embedding_dim = 100
        self.learning_rate =1e-4
        self.n_words = 50000 + 1
        self.checkpoint_step = 1
        self.sentence_length = 1000
        self.cell_layer_size = 1
        self.lambda1 = 0.01  
        #self.path = '/home/santanu/Downloads/Mobile_App/' 
        self.path = path
        self.X_train = np.load(self.path + "aclImdb/X_train.npy") 
        self.y_train = np.load(self.path + "aclImdb/y_train.npy") 
        self.y_train = np.reshape(self.y_train,(-1,1))
        self.X_val = np.load(self.path + "aclImdb/X_val.npy") 
        self.y_val = np.load(self.path + "aclImdb/y_val.npy")
        self.y_val = np.reshape(self.y_val,(-1,1)) 
        self.X_test = np.load(self.path + "aclImdb/X_test.npy") 
        self.y_test = np.load(self.path + "aclImdb/y_test.npy") 
        self.y_test = np.reshape(self.y_test,(-1,1))
        print (np.shape(self.X_train),np.shape(self.y_train))
        print (np.shape(self.X_val),np.shape(self.y_val))
        print (np.shape(self.X_test),np.shape(self.y_test))
        print ('no of positive class in train:',np.sum(self.y_train) )
        print ('no of positive class in test:',np.sum(self.y_val) )   
        self.path_tokenizer = self.path + 'aclImdb/tokenizer.pickle'
        with open(self.path_tokenizer, 'rb') as handle:
            tokenizer = pickle.load(handle)

        self.EMBEDDING_FILE = path + 'glove.6B.100d.txt'
        def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
        embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(self.EMBEDDING_FILE))  
        word_index = tokenizer.word_index
        nb_words = min(self.n_words, len(word_index))
        self.embedding_matrix = np.zeros((self.n_words,self.embedding_dim))
        for word, i in word_index.items():
            if i >= self.n_words: continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None: self.embedding_matrix[i] = embedding_vector 
        
          
          
        
    def _build_model(self):
        
        with tf.variable_scope('inputs'):
            self.X = tf.placeholder(shape=[None, self.sentence_length],dtype=tf.int32,name="X")
            print (self.X)
            self.y = tf.placeholder(shape=[None,1], dtype=tf.float32,name="y")
            self.emd_placeholder = tf.placeholder(tf.float32,shape=[self.n_words,self.embedding_dim]) 

        with tf.variable_scope('embedding'):
            # create embedding variable
            self.emb_W =tf.get_variable('word_embeddings',[self.n_words, self.embedding_dim],initializer=tf.random_uniform_initializer(-1, 1, 0),trainable=True,dtype=tf.float32)
            self.assign_ops = tf.assign(self.emb_W,self.emd_placeholder)
            
            # do embedding lookup
            self.embedding_input = tf.nn.embedding_lookup(self.emb_W,self.X,"embedding_input") 
            print( self.embedding_input )
            self.embedding_input = tf.unstack(self.embedding_input,self.sentence_length,1) 
            #rint( self.embedding_input)

        # define the LSTM cell
        with tf.variable_scope('LSTM_cell'):
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_states)

        
        # define the LSTM operation
        with tf.variable_scope('ops'):
            self.output, self.state = tf.nn.static_rnn(self.cell,self.embedding_input,dtype=tf.float32)
       
       
        with tf.variable_scope('classifier'):
            self.w = tf.get_variable(name="W", shape=[self.hidden_states,1],dtype=tf.float32)
            self.b = tf.get_variable(name="b", shape=[1], dtype=tf.float32)
        self.l2_loss = tf.nn.l2_loss(self.w,name="l2_loss")
        self.scores = tf.nn.xw_plus_b(self.output[-1],self.w,self.b,name="logits")
        self.prediction_probability = tf.nn.sigmoid(self.scores,name='positive_sentiment_probability')
        print (self.prediction_probability)
        self.predictions  = tf.round(self.prediction_probability,name='final_prediction')
            

        self.losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores,labels=self.y)
        self.loss = tf.reduce_mean(self.losses) + self.lambda1*self.l2_loss
        tf.summary.scalar('loss', self.loss)
        
        self.optimizer =  tf.train.AdamOptimizer(self.learning_rate).minimize(self.losses)
        

        self.correct_predictions = tf.equal(self.predictions,tf.round(self.y))
        print (self.correct_predictions)
 
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
        tf.summary.scalar('accuracy', self.accuracy)

    
    def summary(self):
        self.merged = tf.summary.merge_all()
        
    def batch_gen(self,X,y,batch_size):
              
        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        batches = int(X.shape[0]//batch_size)
           
        
        for b in range(batches):
            X_train,y_train = X[index[b*batch_size: (b+1)*batch_size],:],y[index[b*batch_size: (b+1)*batch_size]]
            yield X_train,y_train
        
    def export_model(self,sess,saved_model_dir):
        input_tensor = "inputs/X:0"
        in_text = sess.graph.get_tensor_by_name(input_tensor)
        inputs = {'Review': tf.saved_model.utils.build_tensor_info(in_text)}
        out_probability = sess.graph.get_tensor_by_name('positive_sentiment_probability:0')
        outputs = {'Positive Sentiment Polarity': tf.saved_model.utils.build_tensor_info(out_probability)}
        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,outputs=outputs,method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        
        builder = tf.saved_model.builder.SavedModelBuilder(saved_model_dir)
        builder.add_meta_graph_and_variables(sess,[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                    legacy_init_op=legacy_init_op)
        builder.save() 
        tflite_model  = tf.contrib.lite.toco_convert(sess.graph_def,[self.X[0]],[self.prediction_probability[0]],
				inference_type=1, input_format=1,output_format=2, quantized_input_stats=None, drop_control_dependency=True)
        open(self.path + "converted_model.tflite", "wb").write(tflite_model) 
        
    def _train(self):
        
        self.num_batches = int(self.X_train.shape[0]//self.batch_size)
        self._build_model()
        self.saver = tf.train.Saver()
               
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init) 
            sess.run(self.assign_ops,feed_dict={self.emd_placeholder:self.embedding_matrix})            
            tf.train.write_graph(sess.graph_def, self.path, 'model.pbtxt') 
            print (self.batch_size,self.batch_size_val)
            for epoch in range(self.epochs):
                gen_batch = self.batch_gen(self.X_train,self.y_train,self.batch_size)
                gen_batch_val = self.batch_gen(self.X_val,self.y_val,self.batch_size_val)
                
                for batch in range(self.num_batches):
                    X_batch,y_batch = next(gen_batch) 
                    X_batch_val,y_batch_val = next(gen_batch_val)
                    sess.run(self.optimizer,feed_dict={self.X:X_batch,self.y:y_batch})
                    c,a  = sess.run([self.loss,self.accuracy],feed_dict={self.X:X_batch,self.y:y_batch})
                    print(" Epoch=",epoch," Batch=",batch," Training Loss: ","{:.9f}".format(c), " Training Accuracy=", "{:.9f}".format(a))
                    c1,a1 = sess.run([self.loss,self.accuracy],feed_dict={self.X:X_batch_val,self.y:y_batch_val})
                    print(" Epoch=",epoch," Validation Loss: ","{:.9f}".format(c1), " Validation Accuracy=", "{:.9f}".format(a1))
                results = sess.run(self.prediction_probability,feed_dict={self.X:X_batch_val})
                print(results)

                if epoch % self.checkpoint_step == 0:
                    self.saver.save(sess, os.path.join(self.path,'model'), global_step=epoch)    
            
            self.saver.save(sess,self.path + 'model_ckpt')
            results = sess.run(self.prediction_probability,feed_dict={self.X:X_batch_val})
            print(results)
            
                
                
    def process_main(self):
        self._train()

if __name__ == '__main__':
    with ElapsedTimer('Model train'):
        fire.Fire(Review_sentiment)


    
                
                    
                    
                    
                    
                    
                
                
                

        
        
        


       
        
        
        
        
        
        
        
    
