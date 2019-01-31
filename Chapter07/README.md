# Chapter07: Mobile App for Movie Review Sentiment Analysis
In this modern age, sending data to AI-based applications in the cloud for inference is
commonplace. For instance, a user can send an image taken on a mobile phone to
the Amazon Rekognition API, and the service can tag the various objects, people, text,
scenes, and so on, present in the image. The advantage of employing the service of an AI-
based application that's hosted in the cloud is its ease of use. The mobile app just needs to
make an HTTPS request to the AI-based service, along with the image, and, within a few
seconds, the service will provide the inference results. To decrease latency in response or 
depenency on the internet the user can deploy an app locally on the mobile. In this Chapter we build
such a Movie Review Sentiment Analysis Application deployed on the mobile itself. So all inference 
runs locally on the mobile itself without the need to access the internet and hit a third party service. 


#### Goal 
- [x] To implement a MObile app end to end using Tensorflow capacities
- [x] Understand the technical knowhows of such an application


#### Dataset Link
[Data] (http://ai.stanford.edu/~amaas/data/sentiment/)



#### LSTM model for Classification : Data Preprocessing 

```bash

python preprocess.py --path /home/santanu/Downloads/Mobile_App/aclImdb/

```

#### LSTM model for Classification : Model training

```bash

python movie_review_model_train.py process_main --path /home/santanu/Downloads/Mobile_App/ --epochs 2

```
#### LSTM model for Classification : Freeze the model to a PROTOBUF format

```bash

python freeze_code.py --path /home/santanu/Downloads/Mobile_App/ --MODEL_NAME model

```


#### LSTM model for Classification : Tokenize words 


```bash

python tokenizer_2_txt.py --path '/home/santanu/Downloads/Mobile_App/aclImdb/tokenizer.pickle'  --path_out '/home/santanu/Downloads/Mobile_App/word_ind.txt'

```

#### Code related to the APP

```bash

abc

```

**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 






