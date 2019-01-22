"""
@author: santanu
"""

import numpy as np
import pandas as pd
import argparse
from sklearn.externals import joblib


#path = '/home/santanu/Downloads/RBM Recommender/ml-100k/'
#infile = 'u.data'
#infile_path = path + infile

def process_file(infile_path):
    infile = pd.read_csv(infile_path,sep='\t',header=None)
    infile.columns = ['userId','movieId','rating','timestamp']
    users = list(np.unique(infile.userId.values))
    movies = list(np.unique(infile.movieId.values))
    movies_dict,movies_inverse_dict = {},{}
    for i in range(len(movies)):
        movies_dict[movies[i]] = i
        movies_inverse_dict[i] = movies[i]

    test_data = []
    ratings_matrix = np.zeros([len(users),len(movies),5])
    count = 0
    total_count = len(infile)
    for i in range(len(infile)):
        rec = infile[i:i+1]
        user_index = int(rec['userId']-1)
        movie_index = movies_dict[int(rec['movieId'])]
        rating_index = int(rec['rating']-1)
        if np.random.uniform(0,1) < 0.2 :
            test_data.append([user_index,movie_index,int(rec['rating'])])

        else:
            ratings_matrix[user_index,movie_index,rating_index] = 1

        count +=1
        if (count % 100000 == 0) & (count>= 100000):
            print('Processed ' + str(count) + ' records out of ' + str(total_count))

    np.save(path + 'train_data',ratings_matrix)
    np.save(path + 'test_data',np.array(test_data))
    joblib.dump(movies_dict,path + 'movies_dict.pkl')
    joblib.dump(movies_inverse_dict,path + 'movies_inverse_dict.pkl')
    print(movies_dict)
    print(movies_inverse_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',help='input data path')
    parser.add_argument('--infile',help='input file name')
    args = parser.parse_args()
    path = args.path
    infile = args.infile
    process_file(path + infile)
    
    


