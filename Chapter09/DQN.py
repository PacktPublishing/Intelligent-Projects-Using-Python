import keras
from keras import optimizers
from keras.layers import Convolution2D
from keras.layers import Dense, Flatten, Input, concatenate, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import numpy as np

'''
Double Deep Q Network Implementation
'''

learning_rate = 0.0001
BATCH_SIZE = 128

class DQN:

    def __init__(self,num_states,num_actions,model_path):
        
        self.num_states = num_states
        print(num_states)
        self.num_actions = num_actions
        self.model  = self.build_model()  # Base Model 
        self.model_ = self.build_model()  # target Model (copy of Base Model)

        self.model_chkpoint_1 = model_path +"CarRacing_DDQN_model_1.h5"
        self.model_chkpoint_2 = model_path +"CarRacing_DDQN_model_2.h5"
        
        save_best = keras.callbacks.ModelCheckpoint(self.model_chkpoint_1,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='min',
                                                period=20)
        save_per = keras.callbacks.ModelCheckpoint(self.model_chkpoint_2,
                                                monitor='loss',
                                                verbose=1,
                                                save_best_only=False,
                                                mode='min',
                                                period=400)
        
        self.callbacks_list = [save_best,save_per]
  
    # Convolutional Neural Network that takes in the state and outputs the Q values for all the possible actions.  
    def build_model(self):

        states_in = Input(shape=self.num_states,name='states_in')
        x = Convolution2D(32,(8,8),strides=(4,4),activation='relu')(states_in)
        x = Convolution2D(64,(4,4), strides=(2,2), activation='relu')(x)
        x = Convolution2D(64,(3,3), strides=(1,1), activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(self.num_actions,activation="linear")(x)

        model = Model(inputs=states_in, outputs=x)
        self.opt = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.mse,optimizer=self.opt)
        plot_model(model,to_file='model_architecture.png',show_shapes=True)

        return model
 
    # Train function
    def train(self,x,y,epochs=10,verbose=0):
        self.model.fit(x,y,batch_size=(BATCH_SIZE), epochs=epochs, verbose=verbose, callbacks=self.callbacks_list)
        
   #Predict function
    def predict(self,state,target=False):
        if target:
            # Return the Q value for an action given a state from thr target Network
            return self.model_.predict(state)
        else:
            # Return the Q value from the original Network
            return self.model.predict(state)
    
    # Predict for single state function
    def predict_single_state(self,state,target=False):
        x = state[np.newaxis,:,:,:]
        return self.predict(x,target)
    
    #Update the target Model with the Base Model weights
    def target_model_update(self):
        self.model_.set_weights(self.model.get_weights())
