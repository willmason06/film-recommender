import pandas as pd
import numpy as np
from tqdm import tqdm
import time

from sklearn.linear_model import LinearRegression



'''
optimise hyper parameters
test train cross val

predicted_score weighted with:
popularity_prior
vote_count adjustment
recency penalty

-----add revenue?


'''
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




from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(mean_absolute_error(y_test, pred))



sorted_idx = np.argsort(pred)[::-1]
top_idx = sorted_idx[:5]
top_movie_ids = ids[top_idx]

id_to_title = dict(
    zip(df_tmdb_movies["id"], df_tmdb_movies["title"])
)

top_movies = [
    id_to_title.get(movie_id, "Unknown")
    for movie_id in top_movie_ids
]

print(top_movies)
print(top_movie_ids)
