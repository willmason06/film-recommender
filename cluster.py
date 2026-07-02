import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from sklearn.cluster import SpectralClustering

from visualise import plot_embedding
ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings.npy")
#embeddings = embeddings[:100]


chosen_ids = np.random.choice(ids[:100], size=10, replace=False)

df_tmdb_movies = pd.read_csv("data/tmdb_movie_data.csv")
df_letterboxd_ratings = pd.read_csv('data/letterboxd/ratings_withid.csv')


clustering = SpectralClustering(n_clusters=5,
        assign_labels='discretize',
        random_state=0).fit(embeddings)

plot_embedding('umap', clustering.labels_, chosen_ids)