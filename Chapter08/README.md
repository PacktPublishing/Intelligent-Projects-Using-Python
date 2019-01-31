# Chapter08: Conversational AI Chatbots for Customer Service
Conversational chatbots have produced a lot of hype recently because of their role in
enhancing customer experience. Modern businesses have started using the capabilities of
chatbots in several different processes. Due to the wide acceptance of conversational AIs,
the tedious task of filling out forms or sending information over the internet has become
much more streamlined. One of the desired qualities of a conversational chatbot is that it
should be able to respond to a user request in the current context. The players in a
conversational chatbot system are the user and the bot respectively.


#### Goal 
- [x] To implement a Conversational AI Chatbots for Customer Service
- [x] Understand the technical knowhows of such an application


#### Dataset Link
[Data] (https://www.kaggle.com/thoughtvector/customer-support-on-twitter)


#### Command to train the Chatbot Model

```bash

python chatbot.py --max_vocab_size 50000 --max_seq_len 30 --embedding_dim 100 --hidden_state_dim 100 --epochs 80 --batch_size 128 --learning_rate 1e-4 --data_path /home/santanu/chatbot/data/twcs.csv --outpath /home/santanu/chatbot/ --dropout 0.3 --mode train --num_train_records 50000 --version v1

```

#### Command to perform Inference on the Chatbot Model

```bash

python chatbot.py --max_vocab_size 50000 --max_seq_len 30 --embedding_dim 100 --hidden_state_dim 100 --data_path /home/santanu/chatbot/data/test.csv --outpath /home/santanu/chatbot/ --dropout 0.3 --mode inference --version v1 --load_model_from /home/santanu/chatbot/s2s_model_v1_.h5 --vocabulary_path /home/santanu/chatbot/vocabulary.pkl --reverse_vocabulary_path /home/santanu/chatbot/reverse_vocabulary.pkl --count_vectorizer_path /home/santanu/chatbot/count_vectorizer.pkl

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 






