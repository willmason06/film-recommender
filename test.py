import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# -----------------------------
# Load data
# -----------------------------

ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings.npy")

df_tmdb_movies = pd.read_csv(
    "data/tmdb_movie_data.csv"
)

df_scores = pd.read_csv(
    "data/movie_scores.csv"
)

df_letterboxd_ratings = pd.read_csv(
    "data/letterboxd/ratings_withid.csv"
)


# -----------------------------
# Create lookup tables
# -----------------------------

id_to_index = {
    movie_id: idx
    for idx, movie_id in enumerate(ids)
}

id_to_title = dict(
    zip(
        df_tmdb_movies["id"],
        df_tmdb_movies["title"]
    )
)

score_lookup = {
    row["id"]: (
        row["vote_average"],
        row["vote_count"],
        row["popularity"],
        row["release_date"]
    )
    for _, row in df_scores.iterrows()
}


# -----------------------------
# Build training set
# -----------------------------

X = []
y = []

watched_ids = set()

for movie_id, rating in tqdm(
    zip(
        df_letterboxd_ratings["id"],
        df_letterboxd_ratings["Rating"]
    ),
    total=len(df_letterboxd_ratings)
):

    watched_ids.add(movie_id)

    if movie_id not in id_to_index:
        continue

    idx = id_to_index[movie_id]

    X.append(
        embeddings[idx]
    )

    y.append(
        rating
    )

X = np.array(X)
y = np.array(y)


# -----------------------------
# Train/test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Train model
# -----------------------------

model = Ridge(alpha=1.0)

model.fit(
    X_train,
    y_train
)

pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    pred
)

print()
print(f"MAE: {mae:.3f}")
print()


# -----------------------------
# Predict every movie
# -----------------------------

predicted_ratings = model.predict(
    embeddings
)


# -----------------------------
# Rerank with metadata
# -----------------------------

current_year = 2025

final_scores = np.full(
    len(ids),
    -np.inf
)

for idx, movie_id in enumerate(ids):

    if movie_id in watched_ids:
        continue
    
    if movie_id not in score_lookup:
        continue

    vote_average, vote_count, popularity, release_date = (
        score_lookup[movie_id]
    )

    try:
        year = int(
            str(release_date)[:4]
        )
    except:
        year = 2000

    predicted_rating = predicted_ratings[idx]

    final_score = (
        predicted_rating
        + 0.05 * vote_average
        + 0.2 * np.log1p(vote_count)
        + 0.01 * np.log1p(popularity)
        - 0.002 * (current_year - year)
    )

    final_scores[idx] = final_score


# -----------------------------
# Top recommendations
# -----------------------------

top_idx = np.argsort(
    final_scores
)[::-1][:20]


print("\nTop recommendations:\n")

for idx in top_idx:

    movie_id = ids[idx]

    title = id_to_title.get(
        movie_id,
        "Unknown"
    )

    vote_average, vote_count, popularity, release_date = (
        score_lookup[movie_id]
    )

    print(
        f"{title}"
        f" | score={final_scores[idx]:.2f}"
        f" | tmdb={vote_average:.1f}"
        f" | votes={vote_count}"
        f" | popularity={popularity:.1f}"
        f" | year={str(release_date)[:4]}"
    )