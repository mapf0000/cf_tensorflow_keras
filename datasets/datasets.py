import pandas as pd
import numpy as np

def load_movie_lens_1m():
    movie_lens_1m = pd.read_csv("datasets/ml-1m/ratings.dat", sep='::', header=None, engine='python')
    x, y = movie_lens_1m.iloc[:, :2].values, movie_lens_1m.iloc[:, 2].values 

    return x, y

def load_movie_lens_100k():
    movie_lens_100k = pd.read_csv("datasets/ml-100k/u.data", sep='\t', header=None, engine='python')
    x, y = movie_lens_100k.iloc[:, :2].values, movie_lens_100k.iloc[:, 2].values 

    return x, y