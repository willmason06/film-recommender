import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt


movie_ids = np.load("data/movie_ids.npy")


def plot_embedding(labels=None, highlight_ids=None):
    reduced_embeddings = np.load(
        "data/movie_embeddings_umap2.npy"
    )

    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(reduced_embeddings[:,0], 
               reduced_embeddings[:,1], 
               marker='x', 
               c=labels,
               s=0.5,
               alpha=0.1
               )
    
    if highlight_ids is not None:

        df = pd.read_csv("data/tmdb_movie_data.csv")
        id_to_title = dict(zip(df["id"], df["title"]))
        
        for highlight_id in highlight_ids:
            matches = np.where(movie_ids == highlight_id)[0]

            if len(matches) == 0:
                continue

            idx = matches[0]

            x = reduced_embeddings[idx, 0]
            y = reduced_embeddings[idx, 1]

            title = id_to_title.get(highlight_id, "Unknown")

            ax.scatter(x, y, s=100, c="red", zorder=10)

            ax.text(
                x,
                y,
                title,
                fontsize=10,
                zorder=11
            )
            plt.legend()
                
    
    plt.show()



sample = np.random.choice(movie_ids, size=10)
sample = [31510, 258614, 168513, 342732, 227148]



labels = np.load("data/movie_clusters500.npy")
plot_embedding(labels, sample)


arr = labels


values, counts = np.unique(arr, return_counts=True)

for value, count in zip(values, counts):
    print(value, count)