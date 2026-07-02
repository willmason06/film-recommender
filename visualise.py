import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

embeddings = np.load("data/movie_embeddings.npy")
#embeddings = embeddings[:100]

movie_ids = np.load("data/movie_ids.npy")

df = pd.read_csv("data/tmdb_movie_data.csv")
id_to_title = dict(zip(df["id"], df["title"]))


def plot_embedding(METHOD, labels=None, highlight_id=None):
    match METHOD:
        case 'pca':
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
        
        case 'tsne':
            tsne = TSNE(n_components=2)
            reduced_embeddings = tsne.fit_transform(embeddings)

        case 'umap':
            reduced_embeddings = umap.UMAP().fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], marker='x', c=labels)
    
    if highlight_id is not None:

        for id in highlight_id:

            matches = np.where(movie_ids == id)[0]
            idx = matches[0]

            x = reduced_embeddings[idx, 0]
            y = reduced_embeddings[idx, 1]

            title = id_to_title.get(id, "Unknown")
            ax.scatter(x,y, label=title, s=100)
            


    plt.legend()
    plt.show()