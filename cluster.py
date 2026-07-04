import numpy as np
import hdbscan

# cosine or euclidian clustering?

embeddings = np.load("data/movie_embeddings_umap20.npy")

def cluster(embeddings):

        clusterer = hdbscan.HDBSCAN(
        min_cluster_size=500,
        metric="euclidean",
        gen_min_span_tree=True
        )

        labels = clusterer.fit_predict(embeddings)

        np.save("data/movie_clusters500.npy", labels)


cluster(embeddings)


#from visualise import plot_embedding
#plot_embedding(labels=labels)

