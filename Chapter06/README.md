# Chapter06: The Intelligent Recommender System
With the huge amount of digital information available on the internet, it becomes a
challenge for users to access items efficiently. Recommender systems are information
filtering systems that deal with the problem of digital data overload to pull out items or
information according to the user's preferences, interests, and behavior, inferred from
previous activities.

#### Goal 
- [x] Latent factorization-based collaborative filtering
- [x] Using deep learning for latent factor collaborative filtering
- [x] Using the restricted Boltzmann machine for building recommendation systems
- [x] Learn the technical knowhows

#### Dataset Link
[Data] (https://grouplens.org/datasets/movielens/)
- [x] File name (100K Movie Lens)


#### Notebook for Latent Factor Collaborative Filtering

```bash

Latent Factor Collaborative Filtering using Deep Learning.ipynb 

```

#### Notebook for SVD++ method of Collaborative filtering


```bash

SVD++.ipynb

```
#### Restricted Boltzmann machines for recommendation : Command to Preprocess Ratings

```bash

python preprocess_ratings.py --path '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/ml-100k/' --infile 'u.data'

```


#### Restricted Boltzmann machines for recommendation : Command to Train Model

```bash

python rbm.py main_process --mode train --train_file '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/ml-100k/train_data.npy' --outdir '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/' --num_hidden 5 --epochs 1000

```

#### Restricted Boltzmann machines for recommendation : Command to Evaluate/inference 

```bash

python rbm.py main_process --mode test --train_file '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/pred_all_recs.csv' --test_file '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/ml-100k/test_data.npy' --outdir '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/' --user_info_file '/home/santanu/ML_DS_Catalog-
/Collaborating Filtering/ml-100k/u.user' --movie_info_file '/home/santanu/ML_DS_Catalog-/Collaborating Filtering/ml-100k/u.item'

```


**These are sample commands and need to be changed accordingly based on data repositories,output directory,etc**













 






