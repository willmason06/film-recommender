import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

embeddings = np.load("data/movie_embeddings.npy")
embeddings = embeddings[:100]

def plot_embedding(METHOD, labels):
    match METHOD:
        case 'pca':
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(embeddings)
        
        case 'tsne':
            tsne = TSNE(n_components=2)
            reduced_embeddings = tsne.fit_transform(embeddings)

        case 'umap':
            reduced_embeddings = umap.UMAP().fit_transform(embeddings)

    

    fig, ax = plt.subplots()
    ax.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], marker='x', c=labels)
    plt.show()