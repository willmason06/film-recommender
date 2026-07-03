import numpy as np
import hdbscan

# cosine or euclidian clustering?

def cluster(embeddings):

        clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        metric="euclidean",
        gen_min_span_tree=True
        )

        labels = clusterer.fit_predict(embeddings)

        np.save("data/movie_clusters.npy", labels)





#from visualise import plot_embedding
#plot_embedding(labels=labels)

