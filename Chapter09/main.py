import sys
#sys.path.append('/home/santanu/ML_DS_Catalog-/Python-Artificial-Intelligence-Projects_backup/Python-Artificial-Intelligence-Projects/Chapter09/Scripts/')
from gym import envs
from Agents import Agent,RandomAgent
from helper_functions import action_list,model_save
from environment import environment
import argparse
import numpy as np
import random
from sum_tree import sum_tree
from sklearn.externals import joblib

'''
This is the main module for training and testing the CarRacing Application from gym
'''


if __name__ == "__main__":
    #Define the Parameters for training the Model

    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--environment_name',default='CarRacing-v0')
    parser.add_argument('--model_path',help='model_path')
    parser.add_argument('--train_mode',type=bool,default=True)
    parser.add_argument('--test_mode',type=bool,default=False)
    parser.add_argument('--epsilon_greedy',default=True)
    parser.add_argument('--render',type=bool,default=True)
    parser.add_argument('--width',type=int,default=96)
    parser.add_argument('--height',type=int,default=96)
    parser.add_argument('--num_stack',type=int,default=4)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--huber_loss_thresh',type=float,default=1.)
    parser.add_argument('--dropout',type=float,default=1.)
    parser.add_argument('--memory_size',type=int,default=10000)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--max_num_episodes',type=int,default=500)
    
    
    args = parser.parse_args()
    
    environment_name = args.environment_name
    model_path = args.model_path
    test_mode = args.test_mode
    train_mode = args.train_mode
    epsilon_greedy  = args.epsilon_greedy
    render = args.render
    width = args.width
    height = args.height
    num_stack = args.num_stack
    lr = args.lr
    huber_loss_thresh = args.huber_loss_thresh
    dropout = args.dropout
    memory_size = args.memory_size
    dropout = args.dropout
    batch_size = args.batch_size
    max_num_episodes = args.max_num_episodes
    max_eps = 1
    min_eps = 0.02 
    seed_gym = 2  # Random state
    img_dim = (width,height,num_stack)
    num_actions = len(action_list)

        
if __name__ == '__main__':

    environment_name = 'CarRacing-v0'
    env = environment(environment_name,img_dim,num_stack,num_actions,render,lr)
    num_states  = img_dim
    print(env.env.action_space.shape)
    action_dim = env.env.action_space.shape[0] 
    assert action_list.shape[1] == action_dim,"length of Env action space does not match action buffer"
    num_actions = action_list.shape[0]
    # Setting random seeds with respect to python inbuilt random and numpy random
    random.seed(901)
    np.random.seed(1)
    agent = Agent(num_states, num_actions,img_dim,model_path)
    randomAgent = RandomAgent(num_actions)

    print(test_mode,train_mode)
    
    try:
        #Train agent
        if test_mode:
            if train_mode:
                print("Initialization with random agent. Fill memory")
                while randomAgent.exp < memory_size:
                    env.run(randomAgent)
                    print(randomAgent.exp, "/", memory_size)
        
                agent.memory = randomAgent.memory
                randomAgent = None
                
                print("Starts learning")
                
                while env.episode < max_num_episodes:
                    env.run(agent)
                
                model_save(model_path, "DDQN_model.h5", agent, env.reward)
                
            else:
                # Load train Model 
                
                print('Load pre-trained agent and learn')
                agent.DQN.model.load_weights(model_path+"DDQN_model.h5")
                agent.DQN.target_model_update()
                try :
                    agent.memory = joblib.load(model_path+"DDQN_model.h5"+"Memory")
                    Params = joblib.load(model_path+"DDQN_model.h5"+"agent_param")
                    agent.epsilon = Params[0]
                    agent.steps = Params[1]
                    opt = Params[2]
                    agent.DQN.opt.decay.set_value(opt['decay'])
                    agent.DQN.opt.epsilon = opt['epsilon']
                    agent.DQN.opt.lr.set_value(opt['lr'])
                    agent.DQN.opt.rho.set_value(opt['rho'])
                    env.reward = joblib.load(model_path+"DDQN_model.h5"+"Rewards")
                    del Params, opt
                except:
                    print("Invalid DDQL_Memory_.csv to load")
                    print("Initialization with random agent. Fill memory")
                    while randomAgent.exp < memory_size:
                        env.run(randomAgent)
                        print(randomAgent.exp, "/", memory_size)
            
                    agent.memory = randomAgent.memory
                    randomAgent = None
                
                    agent.maxEpsilone = max_eps/5
                
                print("Starts learning")
                
                while env.episode < max_num_episodes:
                    env.run(agent)
                
                model_save(model_path, "DDQN_model.h5", agent, env.reward)
        else:
            print('Load agent and play')
            agent.DQN.model.load_weights(model_path+"DDQN_model.h5")
            
            done_ctr = 0
            while done_ctr < 5 :
                env.test(agent)
                done_ctr += 1
            
            env.env.close()
    #Graceful exit                          
    except KeyboardInterrupt:
        print('User interrupt..gracefule exit')
        env.env.close()
        
        if test_mode == False:
            # Prompt for Model save
             print('Save model: Y or N?')
             save = input()
             if save.lower() == 'y':
                 model_save(model_path, "DDQN_model.h5", agent, env.reward)
             else:
                print('Model is not saved!')
