from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda"
)

df = pd.read_csv("data/tmdb_movie_data.csv")
df = df.fillna("")

df["genres"] = df["genres"].apply(lambda x: ", ".join(ast.literal_eval(x)))
df["keywords"] = df["keywords"].apply(lambda x: ", ".join(ast.literal_eval(x)))
df["cast"] = df["cast"].apply(lambda x: ", ".join(ast.literal_eval(x)))
df["writers"] = df["writers"].apply(lambda x: ", ".join(ast.literal_eval(x)))

texts = []
ids = []



for movie in df.to_dict("records"):
    movie_text = f"""
    Title: {movie['title']}
    Overview: {movie['overview']}
    Tagline: {movie['tagline']}
    Genres: {movie['genres']}
    Keywords: {movie['keywords']}
    Director: {movie['director']}
    Writers: {movie['writers']}
    Cast: {movie['cast']}
    """

    texts.append(movie_text)
    ids.append(movie["id"])

embeddings = model.encode(
    texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True
)

np.save("data/movie_embeddings.npy", embeddings)
np.save("data/movie_ids.npy", ids)

print("Saved embeddings:", len(ids))