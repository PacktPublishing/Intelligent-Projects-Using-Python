import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding
import numpy as np
import codecs
import argparse
from sklearn.externals import joblib
import pandas as pd
import pickle
from elapsedtimer import ElapsedTimer
# Author - Santanu Pattanayak
# Machine Translation Model - English to French
# Trained on GPU GTX 1070 


class MachineTranslation:

    def __init__(self):
        parser = argparse.ArgumentParser(description='arguments')
        parser.add_argument('--path',help='data file path')
        parser.add_argument('--epochs',type=int,help='Number of epochs to run')
        parser.add_argument('--batch_size',type=int,help='batch size')
        parser.add_argument('--latent_dim',type=int,help='hidden state dimension')
        parser.add_argument('--embedding_dim',type=int,help='embedding dimension')
        parser.add_argument('--num_samples',type=int,help='number of samples to train on')
        parser.add_argument('--outdir',help='number of samples to train on')
        parser.add_argument('--verbose',type=int,help='number of samples to train on',default=1)
        parser.add_argument('--mode',help='train/val',default='train')
        
        
        args = parser.parse_args()
        print(args)
        self.path = args.path
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.latent_dim = args.latent_dim
        self.embedding_dim = args.embedding_dim
        self.num_samples = args.num_samples
        self.outdir = args.outdir

        if args.verbose == 1:
            self.verbose = True
        else:
            self.verbose = False
        self.mode = args.mode 

    def read_input_file(self,path,num_samples=10e13):
        input_texts = []
        target_texts = []
        input_words = set()
        target_words = set()

        
        
        with codecs.open(path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')  # \t as the start of sequence 
            target_text = '\t ' + target_text + ' \n'   # \n as the end  of sequence
            input_texts.append(input_text)
            target_texts.append(target_text)
            for word in input_text.split(" "):
                if word not in input_words:
                    input_words.add(word)
            for word in target_text.split(" "):
                if word not in target_words:
                    target_words.add(word)

        return input_texts,target_texts,input_words,target_words
        

    def vocab_generation(self,path,num_samples,verbose=True):
        
        input_texts,target_texts,input_words,target_words = self.read_input_file(path,num_samples)
        input_words = sorted(list(input_words))
        target_words = sorted(list(target_words))
        self.num_encoder_words = len(input_words)
        self.num_decoder_words = len(target_words)
        self.max_encoder_seq_length = max([len(txt.split(" ")) for txt in input_texts])
        self.max_decoder_seq_length = max([len(txt.split(" ")) for txt in target_texts])

        if verbose == True:
        
            print('Number of samples:', len(input_texts))
            print('Number of unique input tokens:', self.num_encoder_words)
            print('Number of unique output tokens:', self.num_decoder_words)
            print('Max sequence length for inputs:', self.max_encoder_seq_length)
            print('Max sequence length for outputs:', self.max_decoder_seq_length)
        
        self.input_word_index = dict(
            [(word, i) for i, word in enumerate(input_words)])
        self.target_word_index = dict(
            [(word, i) for i, word in enumerate(target_words)])
        self.reverse_input_word_dict = dict(
            (i, word) for word, i in self.input_word_index.items())
        self.reverse_target_word_dict = dict(
            (i, word) for word, i in self.target_word_index.items())
        
   
    def process_input(self,input_texts,target_texts=None,verbose=True):

        encoder_input_data = np.zeros(
            (len(input_texts), self.max_encoder_seq_length),
            dtype='float32')
            
        decoder_input_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length),
            dtype='float32')

        decoder_target_data = np.zeros(
            (len(input_texts), self.max_decoder_seq_length,1),
            dtype='float32')
            
        if self.mode == 'train':
            for i, (input_text, target_text) in enumerate(zip(input_texts,target_texts)):
                for t, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, t] = self.input_word_index[word]
                    except:
                        encoder_input_data[i, t] = self.num_encoder_words  
                        
                for t, word in enumerate(target_text.split(" ")):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                    try:
                        decoder_input_data[i, t] = self.target_word_index[word]
                    except:
                        decoder_input_data[i, t] = self.num_decoder_words 
                    if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    #and will not include the start character.
                        try:
                            decoder_target_data[i, t - 1] = self.target_word_index[word]
                        except:
                            decoder_target_data[i, t - 1] = self.num_decoder_words  
            print(self.num_encoder_words)
            print(self.num_decoder_words)
            print(self.embedding_dim)
            self.english_emb = np.zeros((self.num_encoder_words + 1,self.embedding_dim))
            self.french_emb = np.zeros((self.num_decoder_words + 1,self.embedding_dim))
            return encoder_input_data,decoder_input_data,decoder_target_data,np.array(input_texts),np.array(target_texts)
        else:
            for i, input_text in enumerate(input_texts):
                for t, word in enumerate(input_text.split(" ")):
                    try:
                        encoder_input_data[i, t] = self.input_word_index[word]
                    except:
                        encoder_input_data[i, t] = self.num_encoder_words  

                    


            return encoder_input_data,None,None,np.array(input_texts),None


    def train_test_split(self,num_recs,train_frac=0.8):
        rec_indices = np.arange(num_recs)
        np.random.shuffle(rec_indices)
        train_count = int(num_recs*0.8)
        train_indices =  rec_indices[:train_count]
        test_indices =  rec_indices[train_count:]
        return train_indices,test_indices

    def model_enc_dec(self):
        #Encoder Model
        encoder_inp = Input(shape=(None,),name='encoder_inp')
        encoder_inp1 = Embedding(self.num_encoder_words + 1 ,self.embedding_dim,weights=[self.english_emb])(encoder_inp)
        encoder = LSTM(self.latent_dim, return_state=True,name='encoder')
        encoder_out,state_h, state_c = encoder(encoder_inp1)
        encoder_states = [state_h, state_c]

        #Decoder Model
        decoder_inp = Input(shape=(None,),name='decoder_inp')
        decoder_inp1 = Embedding(self.num_decoder_words +1 ,self.embedding_dim,weights=[self.french_emb])(decoder_inp)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True,name='decoder_lstm')
        decoder_out, _, _ = decoder_lstm(decoder_inp1,initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_words, activation='softmax',name='decoder_dense')
        decoder_out = decoder_dense(decoder_out)
        print(np.shape(decoder_out))
        #Combined Encoder Decoder Model
        model  = Model([encoder_inp, decoder_inp], decoder_out)
        #Encoder Model 
        encoder_model = Model(encoder_inp,encoder_states)
        #Decoder Model
        decoder_inp_h = Input(shape=(self.latent_dim,))
        decoder_inp_c = Input(shape=(self.latent_dim,))
        decoder_inp_state = [decoder_inp_h,decoder_inp_c]
        decoder_out,decoder_out_h,decoder_out_c = decoder_lstm(decoder_inp1,initial_state=decoder_inp_state)
        decoder_out = decoder_dense(decoder_out)
        decoder_out_state = [decoder_out_h,decoder_out_c]
        decoder_model = Model(inputs = [decoder_inp] + decoder_inp_state,output=[decoder_out]+ decoder_out_state)

        return model,encoder_model,decoder_model


    def decode_sequence(self,input_seq,encoder_model,decoder_model):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = self.target_word_index['\t']

        # Sampling loop for a batch of sequences
        stop_condition = False
        decoded_sentence = ''

        while not stop_condition:
            output_word, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_word_index = np.argmax(output_word[0, -1, :])
            try:
                sampled_char = self.reverse_target_word_dict[sampled_word_index]
            except:
                sampled_char = '<unknown>'
            decoded_sentence = decoded_sentence + ' ' + sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
            len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_word_index

            # Update states
            states_value = [h, c]

        return decoded_sentence


# Run training

    def train(self,encoder_input_data,decoder_input_data,decoder_target_data):
        print("Training...")
        
        print(np.shape(encoder_input_data))
        print(np.shape(decoder_input_data))
        print(np.shape(decoder_target_data))

        model,encoder_model,decoder_model = self.model_enc_dec()

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.2)
        # Save model
        model.save(self.outdir + 'eng_2_french_dumm.h5')
        return model,encoder_model,decoder_model

    def inference(self,model,data,encoder_model,decoder_model,in_text):
        in_list,out_list = [],[]
        for seq_index in range(data.shape[0]):

            input_seq = data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq,encoder_model,decoder_model)
            print('-')
            print('Input sentence:', in_text[seq_index])
            print('Decoded sentence:',decoded_sentence)
            in_list.append(in_text[seq_index])
            out_list.append(decoded_sentence)
        return in_list,out_list
    
    def save_models(self,outdir):
        self.model.save(outdir + 'enc_dec_model.h5')
        self.encoder_model.save(outdir + 'enc_model.h5')
        self.decoder_model.save(outdir + 'dec_model.h5')
        
        variables_store = {'num_encoder_words':self.num_encoder_words,
                        'num_decoder_words':self.num_decoder_words,
                        'max_encoder_seq_length':self.max_encoder_seq_length,
                        'max_decoder_seq_length':self.max_decoder_seq_length,
                        'input_word_index':self.input_word_index,
                        'target_word_index':self.target_word_index,
                        'reverse_input_word_dict':self.reverse_input_word_dict,
                        'reverse_target_word_dict':self.reverse_target_word_dict
                        }
        with open(outdir + 'variable_store.pkl','wb') as f:
            pickle.dump(variables_store,f)
            f.close()


    def load_models(self,outdir):
        self.model = keras.models.load_model(outdir + 'enc_dec_model.h5')
        self.encoder_model = keras.models.load_model(outdir + 'enc_model.h5')
        self.decoder_model = keras.models.load_model(outdir + 'dec_model.h5')
        
        with open(outdir + 'variable_store.pkl','rb') as f:
            variables_store = pickle.load(f)
            f.close()

        self.num_encoder_words = variables_store['num_encoder_words']
        self.num_decoder_words = variables_store['num_decoder_words']
        self.max_encoder_seq_length = variables_store['max_encoder_seq_length']
        self.max_decoder_seq_length = variables_store['max_decoder_seq_length']
        self.input_word_index = variables_store['input_word_index']
        self.target_word_index = variables_store['target_word_index']
        self.reverse_input_word_dict = variables_store['reverse_input_word_dict']
        self.reverse_target_word_dict = variables_store['reverse_target_word_dict']
        
    def main(self):

        if self.mode == 'train':
            self.vocab_generation(self.path,self.num_samples,self.verbose) # Generate the vocabulary
            input_texts,target_texts,_,_ = self.read_input_file(self.path,self.num_samples)
            encoder_input_data,decoder_input_data,decoder_target_data,input_texts,target_texts = \
                                                 self.process_input(input_texts,target_texts,True)
            num_recs =  encoder_input_data.shape[0]
            train_indices,test_indices = self.train_test_split(num_recs,0.8)
            encoder_input_data_tr,encoder_input_data_te = encoder_input_data[train_indices,],encoder_input_data[test_indices,]
            decoder_input_data_tr,decoder_input_data_te = decoder_input_data[train_indices,],decoder_input_data[test_indices,]
            decoder_target_data_tr,decoder_target_data_te = decoder_target_data[train_indices,],decoder_target_data[test_indices,]
            input_text_tr,input_text_te = input_texts[train_indices],input_texts[test_indices]                                                      
            self.model,self.encoder_model,self.decoder_model = self.train(encoder_input_data_tr,decoder_input_data_tr,decoder_target_data_tr)
            in_list,out_list = self.inference(self.model,encoder_input_data_te,self.encoder_model,self.decoder_model,input_text_te)
            out_df = pd.DataFrame()
            out_df['English text'] = in_list
            out_df['French text'] = out_list
            out_df.to_csv(self.outdir + 'hold_out_results_validation.csv',index=False)
            self.save_models(self.outdir)
                    
        else:
            self.load_models(self.outdir)
            input_texts,_,_,_ = self.read_input_file(self.path,self.num_samples)
            encoder_input_data,_,_,input_texts,_ = \
                                                 self.process_input(input_texts,'',True)
            in_list,out_list  = self.inference(self.model,encoder_input_data_te,self.encoder_model,self.decoder_model,input_text_te)
            out_df = pd.DataFrame()
            out_df['English text'] = in_list
            out_df['French text'] = out_list
            out_df.to_csv(self.outdir + 'results_test.csv',index=False)

        

        
if __name__ == '__main__':
    obj = MachineTranslation()
    with ElapsedTimer(obj.mode):
        obj.main()


        
            









        



        
