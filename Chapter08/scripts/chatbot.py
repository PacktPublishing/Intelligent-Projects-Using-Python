# __author__ = 'Santanu'

import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, LSTM, Dropout, Embedding, RepeatVector, concatenate,TimeDistributed
from keras.utils import np_utils
from time import time
import argparse
from sklearn.externals import joblib
import re
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import load_model
import pickle 

class chatbot:

    def __init__(self):
        parser = argparse.ArgumentParser(description='Process the inputs')
        parser.add_argument('--max_vocab_size',help='maximum words in the vocabulary',default=50000)
        parser.add_argument('--max_seq_len',help='maximum words to process in tweet',default=30)
        parser.add_argument('--embedding_dim',help='maximum words to process in tweet',default=100)
        parser.add_argument('--hidden_state_dim',help='hidden dimension of the LSTM',default=100)
        parser.add_argument('--epochs',help='Number of epochs to training',default=100)
        parser.add_argument('--batch_size',help='Batch size for training',default=30)
        parser.add_argument('--learning_rate',help='Learning rate for training',default=1e-4)
        parser.add_argument('--dropout',help='dropout',default=None)
        parser.add_argument('--data_path',help='Path for the training dataset')
        parser.add_argument('--outpath',help='output directory')
        parser.add_argument('--version', help='version run of the code',default='v1')
        parser.add_argument('--mode',help='train/inference',default='train')
        parser.add_argument('--num_train_records',help='no of records to use for training',default=20000)
        parser.add_argument('--load_model_from',help='saved model path')
        parser.add_argument('--vocabulary_path',help='vocabulary path')
        parser.add_argument('--reverse_vocabulary_path',help='reverse vocabulary')
        parser.add_argument('--count_vectorizer_path',help='count_vectorizer_path')

         
           
        args = parser.parse_args()

    
        self.max_vocab_size = int(args.max_vocab_size)
        self.max_seq_len = int(args.max_seq_len )
        self.embedding_dim = int(args.embedding_dim)
        self.hidden_state_dim = int(args.hidden_state_dim)
        self.epochs = int(args.epochs)
        self.batch_size = int(args.batch_size)
        self.dropout= float(args.dropout)
        self.learning_rate = float(args.learning_rate)
        self.UNK = 0
        self.PAD = 1
        self.START = 2
        self.data_path = args.data_path
        self.outpath  = args.outpath
        self.version = args.version
        self.mode = args.mode
        self.num_train_records = int(args.num_train_records)
        self.load_model_from = args.load_model_from
        self.vocabulary_path = args.vocabulary_path
        self.reverse_vocabulary_path = args.reverse_vocabulary_path
        self.count_vectorizer_path  = args.count_vectorizer_path

    def process_data(self,path):
        data = pd.read_csv(path)

        if self.mode == 'train':
            data = pd.read_csv(path)
            data['in_response_to_tweet_id'].fillna(-12345,inplace=True)
            tweets_in =  data[data['in_response_to_tweet_id'] == -12345]
            tweets_in_out = tweets_in.merge(data,left_on=['tweet_id'],right_on=['in_response_to_tweet_id'])
            return tweets_in_out[:self.num_train_records]
        elif self.mode == 'inference':
            return data
            
    
    def replace_anonymized_names(self,data):

        def replace_name(match):
            cname = match.group(2).lower()
            if not cname.isnumeric():
                return match.group(1) + match.group(2)
            return '@__cname__'
        
        re_pattern = re.compile('(\W@|^@)([a-zA-Z0-9_]+)')
        #print(data['text_x'])
        if self.mode == 'train':

            in_text = data['text_x'].apply(lambda txt:re_pattern.sub(replace_name,txt))
            out_text = data['text_y'].apply(lambda txt:re_pattern.sub(replace_name,txt))
            return list(in_text.values),list(out_text.values)
        else:
            return list(map(lambda x:re_pattern.sub(replace_name,x),data))

    def tokenize_text(self,in_text,out_text):
        count_vectorizer = CountVectorizer(tokenizer=casual_tokenize, max_features=self.max_vocab_size - 3)
        count_vectorizer.fit(in_text + out_text)
        self.analyzer = count_vectorizer.build_analyzer()
        self.vocabulary = {key_: value_ + 3 for key_,value_ in count_vectorizer.vocabulary_.items()}
        self.vocabulary['UNK'] = self.UNK
        self.vocabulary['PAD'] = self.PAD
        self.vocabulary['START'] = self.START
        self.reverse_vocabulary = {value_: key_ for key_, value_ in self.vocabulary.items()}
        joblib.dump(self.vocabulary,self.outpath + 'vocabulary.pkl')
        joblib.dump(self.reverse_vocabulary,self.outpath + 'reverse_vocabulary.pkl')
        joblib.dump(count_vectorizer,self.outpath + 'count_vectorizer.pkl')
        #pickle.dump(self.count_vectorizer,open(self.outpath + 'count_vectorizer.pkl',"wb"))


    def words_to_indices(self,sent):
        word_indices = [self.vocabulary.get(token,self.UNK) for token in self.analyzer(sent)] + [self.PAD]*self.max_seq_len
        word_indices = word_indices[:self.max_seq_len]
        return word_indices

    def indices_to_words(self,indices):
        return ' '.join(self.reverse_vocabulary[id] for id in indices if id != self.PAD).strip() 

    def data_transform(self,in_text,out_text):
        #rint(in_text),print(out_text)	
        X = [self.words_to_indices(s) for s in in_text]
        Y = [self.words_to_indices(s) for s in out_text]
        #X = lambda sent:self.words_to_indices(sent))
        #Y = out_text.apply(lambda sent:self.words_to_indices(sent))
        return np.array(X),np.array(Y)
    
    def train_test_split_(self,X,Y):
        X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=0)
        #X_train = X_train[:,:,np.newaxis]
        #X_test = X_test[:,:,np.newaxis]
        y_train = y_train[:,:,np.newaxis]
        y_test = y_test[:,:,np.newaxis]	
        return X_train, X_test, y_train, y_test

    
    def data_creation(self):
        data = self.process_data(self.data_path)
        #print(data.head()) 
        in_text,out_text =  self.replace_anonymized_names(data)
        test_sentences = []
        test_indexes= np.random.randint(1,self.num_train_records,10)
        for ind in test_indexes:
            sent = in_text[ind]
            test_sentences.append(sent)
	#print(in_text[:10])
        #rint(in_text.shape,out_text.shape)
        self.tokenize_text(in_text,out_text)
        X,Y = self.data_transform(in_text,out_text)
        X_train, X_test, y_train, y_test = self.train_test_split_(X,Y)
        return X_train, X_test, y_train, y_test,test_sentences

    def define_model(self):
        
        # Embedding Layer
        embedding = Embedding(
            output_dim=self.embedding_dim,
            input_dim=self.max_vocab_size,
            input_length=self.max_seq_len,
            name='embedding',
        )
        
        # Encoder input
    
        encoder_input = Input(
            shape=(self.max_seq_len,),
            dtype='int32',
            name='encoder_input',
        )
        
        embedded_input = embedding(encoder_input)

    
        encoder_rnn = LSTM(
            self.hidden_state_dim,
            name='encoder',
            dropout=self.dropout
        )
        # Context is repeated to the max sequence length so that the same context 
        # can be feed at each step of decoder
        context = RepeatVector(self.max_seq_len)(encoder_rnn(embedded_input))
    
        # Decoder    
        last_word_input = Input(
            shape=(self.max_seq_len,),
            dtype='int32',
            name='last_word_input',
        )
        
        embedded_last_word = embedding(last_word_input)
        # Combines the context produced by the encoder and the last word uttered as inputs
        # to the decoder.
        
        decoder_input = concatenate([embedded_last_word, context],axis=2)

        # return_sequences causes LSTM to produce one output per timestep instead of one at the
        # end of the intput, which is important for sequence producing models.
        decoder_rnn = LSTM(
            self.hidden_state_dim,
            name='decoder',
            return_sequences=True,
            dropout=self.dropout
        )
        
        decoder_output = decoder_rnn(decoder_input)
    
        # TimeDistributed allows the dense layer to be applied to each decoder output per timestep
        next_word_dense = TimeDistributed(
            Dense(int(self.max_vocab_size/20),activation='relu'),
            name='next_word_dense',
        )(decoder_output)
        
        next_word = TimeDistributed(
            Dense(self.max_vocab_size,activation='softmax'),
            name='next_word_softmax'
        )(next_word_dense)
        
        return Model(inputs=[encoder_input,last_word_input], outputs=[next_word])
        
    def create_model(self):
	    _model_ = self.define_model()
	    adam = Adam(lr=self.learning_rate,clipvalue=5.0)
	    _model_.compile(optimizer=adam,loss='sparse_categorical_crossentropy')
	    return _model_

	# Function to append the START index to the response Y
    def include_start_token(self,Y):
            print(Y.shape)
            Y = Y.reshape((Y.shape[0],Y.shape[1]))
            Y = np.hstack((self.START*np.ones((Y.shape[0],1)),Y[:, :-1]))
            #Y = Y[:,:,np.newaxis]	
            return Y
    
    def binarize_output_response(self,Y):
	    return np.array([np_utils.to_categorical(row, num_classes=self.max_vocab_size)
		for row in Y])
    def respond_to_input(self,model,input_sent):
        input_y = self.include_start_token(self.PAD * np.ones((1,self.max_seq_len)))
        ids = np.array(self.words_to_indices(input_sent)).reshape((1,self.max_seq_len))
        for pos in range(self.max_seq_len -1):
            pred = model.predict([ids, input_y]).argmax(axis=2)[0]
            #pred = model.predict([ids, input_y])[0]
            input_y[:,pos + 1] = pred[pos]
        return self.indices_to_words(model.predict([ids,input_y]).argmax(axis=2)[0])
    
    def train_model(self,model,X_train,X_test,y_train,y_test):
        input_y_train = self.include_start_token(y_train)
        print(input_y_train.shape)
        input_y_test = self.include_start_token(y_test)
        print(input_y_test.shape)
        early = EarlyStopping(monitor='val_loss',patience=10,mode='auto')

        checkpoint = ModelCheckpoint(self.outpath + 's2s_model_' + str(self.version) + '_.h5',monitor='val_loss',verbose=1,save_best_only=True,mode='auto')
        lr_reduce = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=2, verbose=0, mode='auto')
        model.fit([X_train,input_y_train],y_train, 
		      epochs=self.epochs,
		      batch_size=self.batch_size, 
		      validation_data=[[X_test,input_y_test],y_test], 
		      callbacks=[early,checkpoint,lr_reduce], 
		      shuffle=True)
        return model

    def generate_response(self,model,sentences):
        output_responses = []
        print(sentences)
        for sent in sentences:
            response = self.respond_to_input(model,sent)
            output_responses.append(response)
        out_df = pd.DataFrame()
        out_df['Tweet in'] = sentences
        out_df['Tweet out'] = output_responses
        return out_df
    
        
    def main(self):
        if self.mode == 'train':

            X_train, X_test, y_train, y_test,test_sentences = self.data_creation()
            print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
            print('Data Creation completed')
            model = self.create_model()
            print("Model creation completed")
            model = self.train_model(model,X_train,X_test,y_train,y_test)
            test_responses = self.generate_response(model,test_sentences)
            print(test_sentences)  
            print(test_responses)
            pd.DataFrame(test_responses).to_csv(self.outpath + 'output_response.csv',index=False)

        elif self.mode == 'inference':

            model = load_model(self.load_model_from)
            self.vocabulary = joblib.load(self.vocabulary_path)
            self.reverse_vocabulary = joblib.load(self.reverse_vocabulary_path)
            #nalyzer_file = open(self.analyzer_path,"rb")
            count_vectorizer = joblib.load(self.count_vectorizer_path)
            self.analyzer = count_vectorizer.build_analyzer()
            data = self.process_data(self.data_path)
            col = data.columns.tolist()[0]
            test_sentences = list(data[col].values)
            test_sentences = self.replace_anonymized_names(test_sentences)
            responses = self.generate_response(model,test_sentences)
            print(responses)
            responses.to_csv(self.outpath + 'responses_' + str(self.version) + '_.csv',index=False)

if __name__ == '__main__':
    start_time = time()
    obj = chatbot()
    obj.main()
    end_time = time()
    print("Processing finished, time taken is %s",end_time - start_time)



