# Chapter09: Autonomous Self-Driving Car though Reinforcement Learning

Reinforcement learning, in which an agent learns to make decisions by interacting with the
environment, has really taken off in the last few years. It is one of the hottest topics in
artificial intelligence and machine learning these days, and research in this domain is
progressing at a fast pace. In reinforcement learning (RL), an agent converts their actions
and experiences into learning to make better decisions in the future. In this Chapter we are 
going to implement an autonomous self-driving car



### Goal 
- [x] implement an autonomous self-driving car using Deep Reinforcement Learning and gym inferface
- [x] Understand the technical knowhows of such an application


#### Dataset Link
Not Applicable  


#### Command to train the Simulated Self Driving Car

```bash
python main.py --environment_name 'CarRacing-v0' --model_path '/home/santanu/Autonomous Car/train/' --train_mode True --test_mode False --epsilon_greedy True --render True --width 96 --height 96 --num_stack 4 --huber_loss_thresh 1 --dropout 0.2 --memory_size 10000 --batch_size 128 --max_num_episodes 500

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 






