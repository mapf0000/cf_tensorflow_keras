import pandas as pd
import numpy as np

def load_movie_lens_1m():
    movie_lens_1m = pd.read_csv("datasets/ml-1m/ratings.dat", sep='::', header=None, engine='python')
        
    x, y = movie_lens_1m.iloc[:, :2].values, movie_lens_1m.iloc[:, 2].values 
    user_dict = dict(enumerate(np.unique(x[:, 0])))
    item_dict = dict(enumerate(np.unique(x[:, 1])))
    x[:, 0] = [{value:key for key, value in user_dict.items()}[u] for u in x[:, 0]] # index users from 0 to num_users - 1
    x[:, 1] = [{value:key for key, value in item_dict.items()}[i] for i in x[:, 1]] # index items from 0 to num_items - 1

    return x, y, user_dict, item_dict

def load_movie_lens_100k():
    movie_lens_100k = pd.read_csv("datasets/ml-100k/u.data", sep='\t', header=None, engine='python')
        
    x, y = movie_lens_100k.iloc[:, :2].values, movie_lens_100k.iloc[:, 2].values 
    #user_dict = dict(enumerate(np.unique(x[:, 0])))
    #item_dict = dict(enumerate(np.unique(x[:, 1])))
    #x[:, 0] = [{value:key for key, value in user_dict.items()}[u] for u in x[:, 0]] # index users from 0 to num_users - 1
    #x[:, 1] = [{value:key for key, value in item_dict.items()}[i] for i in x[:, 1]] # index items from 0 to num_items - 1

    return x, y#, user_dict, item_dict