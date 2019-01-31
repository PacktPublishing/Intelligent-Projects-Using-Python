import gym
from gym import envs
import numpy as np
from helper_functions import rgb2gray,action_list,sel_action,sel_action_index
from keras import backend as K 

seed_gym = 3
action_repeat_num = 8
patience_count = 200
epsilon_greedy = True
max_reward =  10
grass_penalty  = 0.8
max_num_steps = 200 
max_num_episodes = action_repeat_num*100

'''
Enviroment to interact with the Agent
'''

class environment:
    
    def __init__(self, environment_name,img_dim,num_stack,num_actions,render,lr):
        self.environment_name = environment_name
        print(self.environment_name)
        self.env = gym.make(self.environment_name)
        envs.box2d.car_racing.WINDOW_H = 500
        envs.box2d.car_racing.WINDOW_W = 600
        self.episode = 0
        self.reward = []    
        self.step = 0
        self.stuck_at_local_minima = 0
        self.img_dim = img_dim
        self.num_stack = num_stack
        self.num_actions = num_actions
        self.render = render
        self.lr = lr
        if self.render == True:
            print("Rendering proeprly set")
        else:
            print("issue in Rendering")
             
    # Agent performing its task    
    def run(self,agent):
        self.env.seed(seed_gym)              
        img = self.env.reset()
        img =  rgb2gray(img, True)
        s = np.zeros(self.img_dim)
        #Collecting the state
        for i in range(self.num_stack):
            s[:,:,i] = img

            
        s_ = s 
        R = 0
        self.step = 0
                        
        a_soft = a_old = np.zeros(self.num_actions)
        a = action_list[0]
        #print(agent.agent_type)
        while True: 
            if agent.agent_type == 'Learning' : 
                if self.render == True :
                    self.env.render("human")
 

            if self.step % action_repeat_num == 0:
                
                if agent.rand == False:
                    a_old = a_soft
                
                #Agent outputs the action
                a,a_soft = agent.act(s)
                
                # Rescue Agent stuck at local minima
                if epsilon_greedy:
                    if agent.rand == False:
                        if a_soft.argmax() == a_old.argmax():
                            self.stuck_at_local_minima += 1
                            if self.stuck_at_local_minima >= patience_count:
                                print('Stuck in local minimum, reset learning rate')
                                agent.steps = 0
                                K.set_value(agent.DQN.opt.lr,self.lr*10)
                                self.stuck_at_local_minima = 0
                        else:
                            self.stuck_at_local_minima = max(self.stuck_at_local_minima -2, 0)
                            K.set_value(agent.DQN.opt.lr,self.lr)
                #Perform the action on the environment        
                img_rgb, r,done,info = self.env.step(a)
            
                if not done:
                    # Create the next state
                    img =  rgb2gray(img_rgb, True)
                    for i in range(self.num_stack-1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:,self.num_stack-1] = img
            
                else:
                   s_ = None
                # Cumulative reward tracking  
                R += r
                # Normalize reward given by the gym environment 
                r = (r/max_reward) 
                if np.mean(img_rgb[:,:,1]) > 185.0:
                # Penalize if the car is on the grass
                    r -= grass_penalty   
                # Keeping the value of reward within -1 and 1 
                r = np.clip(r, -1 ,1)
                #Agent has a whole state,action,reward,and next state to learb from
                agent.observe( (s, a, r, s_) )
                agent.replay()            
                s = s_
            
            else:
                img_rgb, r, done, info = self.env.step(a)
                if not done:
                    
                    img =  rgb2gray(img_rgb, True)
                    for i in range(self.num_stack-1):
                        s_[:,:,i] = s_[:,:,i+1]
                    s_[:,:,self.num_stack-1] = img
                else:
                   s_ = None

                R += r
                s = s_
                
            if (self.step % (action_repeat_num * 5) == 0) and (agent.agent_type=='Learning'):
                print('step:', self.step, 'R: %.1f' % R, a, 'rand:', agent.rand)
            
            self.step += 1
            
            if done or (R <-5) or (self.step > max_num_steps) or np.mean(img_rgb[:,:,1]) > 185.1:
                self.episode += 1
                self.reward.append(R)
                print('Done:', done, 'R<-5:', (R<-5), 'Green >185.1:',np.mean(img_rgb[:,:,1]))
                break

        print("Episode ",self.episode,"/", max_num_episodes,agent.agent_type) 
        print("Average Episode Reward:", R/self.step, "Total Reward:", sum(self.reward))
    
    
    def test(self,agent):
        self.env.seed(seed_gym)
        img= self.env.reset()
        img = rgb2gray(img, True)
        s = np.zeros(self.img_dim)
        for i in range(self.num_stack):
            s[:,:,i] = img

        R = 0
        self.step = 0
        done = False
        while True :
            self.env.render('human')
                            
            if self.step % action_repeat_num == 0:
                if(agent.agent_type == 'Learning'):
                    act1 = agent.DQN.predict_single_state(s)
                    act = sel_action(np.argmax(act1))
                else:
                    act = agent.act(s)
                    
                if self.step <= 8:
                    act = sel_action(3)
                    
                img_rgb, r, done,info = self.env.step(act)
                img = rgb2gray(img_rgb, True)
                R += r
        
                for i in range(self.num_stack-1):
                    s[:,:,i] = s[:,:,i+1]
                s[:,:,self.num_stack-1] = img
            
            if(self.step % 10) == 0:
                print('Step:', self.step, 'action:',act, 'R: %.1f' % R)
                print(np.mean(img_rgb[:,:,0]), np.mean(img_rgb[:,:,1]), np.mean(img_rgb[:,:,2]))
            self.step += 1
            
            if done or (R< -5) or (agent.steps > max_num_steps) or np.mean(img_rgb[:,:,1]) > 185.1:
                R = 0
                self.step = 0
                print('Done:', done, 'R<-5:', (R<-5), 'Green> 185.1:',np.mean(img_rgb[:,:,1]))
                break
