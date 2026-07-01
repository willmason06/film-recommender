import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from sklearn.linear_model import LinearRegression

ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings.npy")

df_tmdb_movies = pd.read_csv("data/tmdb_movie_data.csv")
df_letterboxd_ratings = pd.read_csv('data/letterboxd/ratings_withid.csv')

watched_ids = df_letterboxd_ratings['id']
watched_ratings = df_letterboxd_ratings['Rating']

def get_index_from_tmdb_id(movie_id):
    matches = np.where(ids == movie_id)[0]

    if len(matches) == 0:
        print(f"Movie ID {movie_id} not found in dataset")
        return 'abcde'

    return matches[0]

# letterboxd -> id, embedding (x), rating (y) 

X = []
y = []

for id, rating in tqdm(zip(watched_ids, watched_ratings),
    total=len(watched_ids)):

    movie_index = get_index_from_tmdb_id(id)
    if movie_index == 'abcde':
        continue
    embedding = embeddings[movie_index]

    X.append(embedding)
    y.append(rating)

X = np.array(X)
y = np.array(y)




model = LinearRegression()
model.fit(X, y)
predicted_ratings = model.predict(embeddings)



sorted_idx = np.argsort(predicted_ratings)[::-1]
top_idx = sorted_idx[:5]
titles = df_tmdb_movies['title']
top_movies = titles[top_idx]
print(top_movies)
