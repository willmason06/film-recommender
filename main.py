import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ids = np.load("data/movie_ids.npy")
embeddings = np.load("data/movie_embeddings.npy")
df = pd.read_csv("data/tmdb_movie_data.csv")


id_to_title = dict(zip(df["id"], df["title"]))

def get_index_from_tmdb_id(movie_id):
    matches = np.where(ids == movie_id)[0]

    if len(matches) == 0:
        raise ValueError(f"Movie ID {movie_id} not found in dataset")

    return matches[0]


def get_most_similar(movie_index, top_k=5):
    query_vec = embeddings[movie_index].reshape(1, -1)

    scores = cosine_similarity(query_vec, embeddings)[0]

    sorted_idx = np.argsort(scores)[::-1]

    sorted_idx = sorted_idx[1:top_k+1]

    query_id = ids[movie_index]
    query_title = id_to_title.get(query_id, "Unknown")

    print(f"\nInput movie: {query_title} (TMDB ID: {query_id})\n")
    print("Most similar movies:\n")

    for i in sorted_idx:
        movie_id = ids[i]
        title = id_to_title.get(movie_id, "Unknown")

        print(f"{title} ({movie_id}) | similarity: {scores[i]:.4f}")


def comparison(movie_index1, movie_index2, top_k=5):
    query_vec1 = embeddings[movie_index1].reshape(1, -1)
    query_vec2 = embeddings[movie_index2].reshape(1, -1)

    combined_vec = (query_vec1 + query_vec2) / 2
    scores = cosine_similarity(combined_vec, embeddings)[0]

    sorted_idx = np.argsort(scores)[::-1]

    sorted_idx = sorted_idx[:top_k+1]

    # what the hell -- why not skip this
    query_id1 = ids[movie_index1]
    query_id2 = ids[movie_index2]
    # what the hell
    query_title1 = id_to_title.get(query_id1, "Unknown")
    query_title2 = id_to_title.get(query_id2, "Unknown")

    print(f"\nInput movies: {query_title1} and {query_title2} \n")
    print("Most similar movies:\n")

    for i in sorted_idx:
        movie_id = ids[i]
        title = id_to_title.get(movie_id, "Unknown")

        print(f"{title} ({movie_id}) | similarity: {scores[i]:.4f}")


tmdb_id1 = 577922
tmdb_id2 = 346698



movie_index1 = get_index_from_tmdb_id(tmdb_id1)
movie_index2 = get_index_from_tmdb_id(tmdb_id2)

#comparison(movie_index1, movie_index2)

get_most_similar(movie_index2)