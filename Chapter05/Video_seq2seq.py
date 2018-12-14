#-*- coding: utf-8 -*-
__author__ = "Santanu Pattanayak"

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import ipdb
import time
import cv2
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
#from  tf.contrib.rnn import core_rnn_cell
from tensorflow.contrib import rnn
print tf.__version__
class video_caption:
    
    
    def __init__(self,cnn_feat_dim,h_dim,batch_size,lstm_steps,video_steps,out_steps,learning_rate,epochs,frame_step):
        self.dim_image = cnn_feat_dim
        self.dim_hidden = h_dim
        self.batch_size = batch_size
        self.lstm_steps = lstm_steps
        self.video_lstm_step=video_steps
        self.caption_lstm_step=out_steps
        self.bias_init_vector=None
        self.path_prj = '/home/santanu/Downloads/Video Captioning/' 
        self.train_text_path = self.path_prj + "video_corpus.csv"
        self.train_feat_path = self.path_prj + "rgb_feats/"

        self.test_text_path = self.path_prj + "video_corpus_test.csv"
        self.test_feat_path = self.path_prj + "rgb_feats/"
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.frame_step = frame_step

    def build_model(self):
        with tf.device("/cpu:0"):
            self.word_emb = tf.Variable(tf.random_uniform([self.n_words, self.dim_hidden], -0.1, 0.1), name='word_emb')


         #   self.lstm1 = rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
         #   self.lstm2 = rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
            self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
	    self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
  

            self.encode_W = tf.Variable( tf.random_uniform([self.dim_image,self.dim_hidden], -0.1, 0.1), name='encode_W')
            self.encode_b = tf.Variable( tf.zeros([self.dim_hidden]), name='encode_b')

            self.word_emb_W = tf.Variable(tf.random_uniform([self.dim_hidden,self.n_words], -0.1,0.1), name='word_emb_W')
            if self.bias_init_vector is not None:
                self.word_emb_b = tf.Variable(self.bias_init_vector.astype(np.float32), name='word_emb_b')
            else:
                self.word_emb_b = tf.Variable(tf.zeros([self.n_words]), name='word_emb_b')

        video = tf.placeholder(tf.float32, [self.batch_size, self.video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_W,self.encode_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.lstm_steps, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.caption_lstm_step): ## Phase 2 => only generate captions
            #if i == 0:
            #    current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            #else:
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.word_emb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [current_embed, output1]), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.word_emb_W, self.word_emb_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs


    def build_generator(self):
        with tf.device("/cpu:0"):
            self.word_emb = tf.Variable(tf.random_uniform([self.n_words, self.dim_hidden], -0.1, 0.1), name='word_emb')


         #   self.lstm1 = rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
         #   self.lstm2 = rnn.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
            self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)
            self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, state_is_tuple=False)


            self.encode_W = tf.Variable( tf.random_uniform([self.dim_image,self.dim_hidden], -0.1, 0.1), name='encode_W')
            self.encode_b = tf.Variable( tf.zeros([self.dim_hidden]), name='encode_b')

            self.word_emb_W = tf.Variable(tf.random_uniform([self.dim_hidden,self.n_words], -0.1,0.1), name='word_emb_W')
            if self.bias_init_vector is not None:
                self.word_emb_b = tf.Variable(self.bias_init_vector.astype(np.float32), name='word_emb_b')
            else:
                self.word_emb_b = tf.Variable(tf.zeros([self.n_words]), name='word_emb_b')


        video = tf.placeholder(tf.float32, [1, self.video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_W, self.encode_b)
        image_emb = tf.reshape(image_emb, [1, self.video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(0, self.video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [padding, output1]), state2)

        for i in range(0, self.caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.word_emb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat(1, [current_embed, output1]), state2)

            logit_words = tf.nn.xw_plus_b( output2, self.word_emb_W, self.word_emb_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.word_emb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds


    def get_train_data(self,text_path,feat_path):
        text_data = pd.read_csv(text_path, sep=',')
        text_data = text_data[text_data['Language'] == 'English']
        text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
        text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
        text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists(x))]
        text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]
        
        unique_filenames = sorted(text_data['video_path'].unique())
        train_data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
        return train_data
        
    
    def get_test_data(self,text_path,feat_path):
        text_data = pd.read_csv(text_path, sep=',')
        text_data = text_data[text_data['Language'] == 'English']
        text_data['video_path'] = text_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End']))+'.npy', axis=1)
        text_data['video_path'] = text_data['video_path'].map(lambda x: os.path.join(feat_path, x))
        text_data = text_data[text_data['video_path'].map(lambda x: os.path.exists( x ))]
        text_data = text_data[text_data['Description'].map(lambda x: isinstance(x, str))]
    
        unique_filenames = sorted(text_data['video_path'].unique())
        test_data = text_data[text_data['video_path'].map(lambda x: x in unique_filenames)]
        return test_data
        
    
    def create_word_dict(self,sentence_iterator, word_count_threshold=5):
        
        word_counts = {}
        sent_cnt = 0
        
        for sent in sentence_iterator:
            sent_cnt += 1
            for w in sent.lower().split(' '):
               word_counts[w] = word_counts.get(w, 0) + 1
        vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
        
        idx2word = {}
        idx2word[0] = '<pad>'
        idx2word[1] = '<bos>'
        idx2word[2] = '<eos>'
        idx2word[3] = '<unk>'
    
        word2idx = {}
        word2idx['<pad>'] = 0
        word2idx['<bos>'] = 1
        word2idx['<eos>'] = 2
        word2idx['<unk>'] = 3
    
        for idx, w in enumerate(vocab):
            word2idx[w] = idx+4
            idx2word[idx+4] = w
    
        word_counts['<pad>'] = sent_cnt
        word_counts['<bos>'] = sent_cnt
        word_counts['<eos>'] = sent_cnt
        word_counts['<unk>'] = sent_cnt
    
        bias_init_vector = np.array([1.0 * word_counts[ idx2word[i] ] for i in idx2word])
        bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
        bias_init_vector = np.log(bias_init_vector)
        bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
        return word2idx,idx2word,bias_init_vector
        
        
    
        
        
    
    def train(self):
        self.train_data = self.get_train_data(self.train_text_path,self.train_feat_path)
        train_captions = self.train_data['Description'].values
        self.test_data = self.get_test_data(self.test_text_path,self.test_feat_path)
        test_captions = self.test_data['Description'].values
    
        captions_list = list(train_captions) + list(test_captions)
        captions = np.asarray(captions_list, dtype=np.object)
    
        captions = map(lambda x: x.replace('.', ''), captions)
        captions = map(lambda x: x.replace(',', ''), captions)
        captions = map(lambda x: x.replace('"', ''), captions)
        captions = map(lambda x: x.replace('\n', ''), captions)
        captions = map(lambda x: x.replace('?', ''), captions)
        captions = map(lambda x: x.replace('!', ''), captions)
        captions = map(lambda x: x.replace('\\', ''), captions)
        captions = map(lambda x: x.replace('/', ''), captions)
    
        self.word2idx,self.idx2word,self.bias_init_vector = self.create_word_dict(captions, word_count_threshold=0)
        
        np.save(self.path_prj + "word2idx",self.word2idx)
        np.save(self.path_prj + "idx2word" ,self.idx2word)
        np.save(self.path_prj + "bias_init_vector",self.bias_init_vector)
        self.n_words = len(self.word2idx)
    
        tf_loss, tf_video,tf_video_mask,tf_caption,tf_caption_mask, tf_probs = self.build_model()
        sess = tf.InteractiveSession()
        
        saver = tf.train.Saver(max_to_keep=100, write_version=1)
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(tf_loss)
        tf.global_variables_initializer().run()
    
        #new_saver = tf.train.Saver()
        #new_saver = tf.train.import_meta_graph('./rgb_models/model-1000.meta')
        #new_saver.restore(sess, tf.train.latest_checkpoint('./models/'))
    
        loss_out = open('loss.txt', 'w')
        val_loss = []
    
        for epoch in range(0,self.epochs):
            val_loss_epoch = []
    
            index = list(self.train_data.index)
            np.random.shuffle(index)
            self.train_data = self.train_data.ix[index]
    
            current_train_data = self.train_data.groupby('video_path').apply(lambda x: x.irow(np.random.choice(len(x))))
            current_train_data = current_train_data.reset_index(drop=True)
    
            for start, end in zip(
                    range(0, len(current_train_data),batch_size),
                    range(batch_size,len(current_train_data),batch_size)):
    
                start_time = time.time()
    
                current_batch = current_train_data[start:end]
                current_videos = current_batch['video_path'].values
    
                current_feats = np.zeros((self.batch_size, self.video_lstm_step,self.dim_image))
                current_feats_vals = map(lambda vid: np.load(vid),current_videos)
    
                current_video_masks = np.zeros((self.batch_size,self.video_lstm_step))
    
                for ind,feat in enumerate(current_feats_vals):
                    current_feats[ind][:len(current_feats_vals[ind])] = feat
                    current_video_masks[ind][:len(current_feats_vals[ind])] = 1
    
                current_captions = current_batch['Description'].values
                current_captions = map(lambda x: '<bos> ' + x, current_captions)
                current_captions = map(lambda x: x.replace('.', ''), current_captions)
                current_captions = map(lambda x: x.replace(',', ''), current_captions)
                current_captions = map(lambda x: x.replace('"', ''), current_captions)
                current_captions = map(lambda x: x.replace('\n', ''), current_captions)
                current_captions = map(lambda x: x.replace('?', ''), current_captions)
                current_captions = map(lambda x: x.replace('!', ''), current_captions)
                current_captions = map(lambda x: x.replace('\\', ''), current_captions)
                current_captions = map(lambda x: x.replace('/', ''), current_captions)
    
                for idx, each_cap in enumerate(current_captions):
                    word = each_cap.lower().split(' ')
                    if len(word) < self.caption_lstm_step:
                        current_captions[idx] = current_captions[idx] + ' <eos>'
                    else:
                        new_word = ''
                        for i in range(self.caption_lstm_step-1):
                            new_word = new_word + word[i] + ' '
                        current_captions[idx] = new_word + '<eos>'
    
                current_caption_ind = []
                for cap in current_captions:
                    current_word_ind = []
                    for word in cap.lower().split(' '):
                        if word in self.word2idx:
                            current_word_ind.append(self.word2idx[word])
                        else:
                            current_word_ind.append(self.word2idx['<unk>'])
                    current_caption_ind.append(current_word_ind)
    
                current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=self.caption_lstm_step)
                current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
                current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
                nonzeros = np.array( map(lambda x: (x != 0).sum() + 1, current_caption_matrix ) )
    
                for ind, row in enumerate(current_caption_masks):
                    row[:nonzeros[ind]] = 1
    
                probs_val = sess.run(tf_probs, feed_dict={
                    tf_video:current_feats,
                    tf_caption: current_caption_matrix
                    })
    
                _, loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_video: current_feats,
                            tf_video_mask : current_video_masks,
                            tf_caption: current_caption_matrix,
                            tf_caption_mask: current_caption_masks
                            })
                val_loss_epoch.append(loss_val)
    
                print 'idx: ', start, " Epoch: ", epoch, " loss: ", loss_val, ' Elapsed time: ', str((time.time() - start_time))
                loss_out.write('epoch ' + str(epoch) + ' loss ' + str(loss_val) + '\n')
    
            # draw loss curve every epoch
            val_loss.append(np.mean(val_loss_epoch))
            plt_save_dir = self.path_prj + "loss_imgs"
            plt_save_img_name = str(epoch) + '.png'
            plt.plot(range(len(val_loss)),val_loss, color='g')
            plt.grid(True)
            plt.savefig(os.path.join(plt_save_dir, plt_save_img_name))
    
            if np.mod(epoch, 10) == 0:
                print "Epoch ", epoch, " is done. Saving the model ..."
                saver.save(sess, os.path.join(self.path_prj, 'model'), global_step=epoch)
    
        loss_out.close()
        
        
    
    def inference_(self,model_path='./models/model-100'):
        self.test_data = self.get_test_data(self.test_text_path,self.test_feat_path)
        test_videos = self.test_data['video_path'].unique()
    
        self.idx2word = pd.Series(np.load(self.path_prj + "idx2word.npy").tolist())
    
        self.bias_init_vector = np.load(self.path_prj + "bias_init_vector.npy")
    
       
        self.n_words = len(self.idx2word)
        video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = self.build_generator()
    
        sess = tf.InteractiveSession()
    
        saver = tf.train.Saver()
        saver.restore(sess,model_path)
    
        test_output_txt_fd = open('S2VT_results.txt', 'w')
        for idx, video_feat_path in enumerate(test_videos):
            print idx,video_feat_path
            video_feat = np.load(video_feat_path)[None,...]
            if video_feat.shape[1] == self.frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            else:
                continue
                #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
                #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
                #video_feat = shape_templete
                #video_mask = np.ones((video_feat.shape[0], n_frame_step))
    
            gen_word_idx = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
            gen_words = self.idx2word[gen_word_idx]
    
            punct = np.argmax(np.array(gen_words) == '<eos>') + 1
            gen_words = gen_words[:punct]
    
            gen_sent = ' '.join(gen_words)
            gen_sent = gen_sent.replace('<bos> ', '')
            gen_sent = gen_sent.replace(' <eos>', '')
            print gen_sent,'\n'
            test_output_txt_fd.write(video_feat_path + '\n')
            test_output_txt_fd.write(gen_sent + '\n\n')
        

if __name__ == '__main__':
    
    cnn_feat_dim = 4096
    h_dim = 512  
    batch_size = 8
    video_steps=80
    lstm_steps = 80 
    out_steps=20
    bias_init_vector=None
    learning_rate = 1e-4
    epochs = 100
    frame_step = 80
        
    if sys.argv[1] == 'train':
        model = video_caption(cnn_feat_dim,h_dim,batch_size,lstm_steps,video_steps,out_steps,learning_rate,epochs,frame_step)
        model.train()
    
    if sys.argv[1] == 'test':
        model = video_caption(cnn_feat_dim,h_dim,batch_size,lstm_steps,video_steps,out_steps,learning_rate,epochs,frame_step)
        model.inference_(model_path='/home/santanu/Downloads/Video Captioning/model-90')
        
      
        
        

