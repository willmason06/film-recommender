import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load data
# -----------------------------
ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings.npy")

df = pd.read_csv("data/tmdb_movie_data.csv")

# map id -> title
id_to_title = dict(zip(df["id"], df["title"]))

# -----------------------------
# convert TMDB ID → array index
# -----------------------------
def get_index_from_tmdb_id(movie_id):
    matches = np.where(ids == movie_id)[0]

    if len(matches) == 0:
        raise ValueError(f"Movie ID {movie_id} not found in dataset")

    return matches[0]

# -----------------------------
# similarity function
# -----------------------------
def get_most_similar(movie_index, top_k=5):
    query_vec = embeddings[movie_index].reshape(1, -1)

    scores = cosine_similarity(query_vec, embeddings)[0]

    sorted_idx = np.argsort(scores)[::-1]

    # remove itself
    sorted_idx = sorted_idx[1:top_k+1]

    query_id = ids[movie_index]
    query_title = id_to_title.get(query_id, "Unknown")

    print(f"\nInput movie: {query_title} (TMDB ID: {query_id})\n")
    print("Most similar movies:\n")

    for i in sorted_idx:
        movie_id = ids[i]
        title = id_to_title.get(movie_id, "Unknown")

        print(f"{title} ({movie_id}) | similarity: {scores[i]:.4f}")

# -----------------------------
# interactive loop (TMDB ID input)
# -----------------------------
while True:
    user_input = input("\nEnter TMDB movie ID (-1 to quit): ")

    if user_input == "-1":
        break

    try:
        tmdb_id = int(user_input)

        movie_index = get_index_from_tmdb_id(tmdb_id)
        get_most_similar(movie_index)

    except Exception as e:
        print("Error:", e)