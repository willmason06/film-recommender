import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt


movie_ids = np.load("data/movie_ids.npy")


def plot_embedding(labels=None, highlight_id=None):
    reduced_embeddings = np.load(
        "data/movie_embeddings_umap.npy"
    )

    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(reduced_embeddings[:,0], 
               reduced_embeddings[:,1], 
               marker='x', 
               c=labels,
               s=0.5,
               alpha=0.1
               )
    
    if highlight_id is not None:

        df = pd.read_csv("data/tmdb_movie_data.csv")
        id_to_title = dict(zip(df["id"], df["title"]))

        title = id_to_title.get(id, "Unknown")

        matches = np.where(movie_ids == id)[0]
        idx = matches[0]

        x = reduced_embeddings[idx, 0]
        y = reduced_embeddings[idx, 1]


        # fix this
        title = id_to_title.get(highlight_id, "Unknown")

        ax.scatter(
            x,
            y,
            s=100,
            c="red"
        )

        ax.text(
            x,
            y,
            title,
            fontsize=10
        )
            
    plt.legend()
    plt.show()

    