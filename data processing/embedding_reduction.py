import numpy as np
from sklearn.decomposition import PCA
import umap
import os

# save the umap config for if there needs to be any reruns

if __name__ == "__main__":
        
    if os.path.exists("data/movie_embeddings_pca50.npy"):

        reduced_embeddings50 = np.load(
            "data/movie_embeddings_pca50.npy"
        )
        print('file50 exists')

    else:
        embeddings = np.load(
            "data/movie_embeddings.npy"
        ).astype(np.float32)

        pca = PCA(
        n_components=50,
        random_state=42
        )

        reduced_embeddings50 = pca.fit_transform(embeddings).astype(np.float32)

        np.save(
                "data/movie_embeddings_pca50.npy",
                reduced_embeddings50
                )

    if os.path.exists("data/movie_embeddings_umap20.npy"):
        
        reduced_embeddings20 = np.load(
            "data/movie_embeddings_umap20.npy"
        )
        print('file20 exists')

    else:
            
        reducer20 = umap.UMAP(
        n_components=20,
        metric="cosine",
        random_state=42,
        n_neighbors=100,
        verbose=True
        )

        reduced_embeddings20 = reducer20.fit_transform(reduced_embeddings50)

        np.save(
        "data/movie_embeddings_umap20.npy",
        reduced_embeddings20
        )

        print('reduced 20 saved')


    if os.path.exists("data/movie_embeddings_umap2.npy"):
        
        reduced_embeddings2 = np.load(
            "data/movie_embeddings_umap2.npy"
        )
        print('file2 exists')

    else:

        reducer2 = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=30,
        min_dist=0.05,
        random_state=42,
        verbose=True
        )

        reduced_embeddings2 = reducer2.fit_transform(reduced_embeddings50)

        np.save(
        "data/movie_embeddings_umap2.npy",
        reduced_embeddings2
        )

        print('reduced 2 saved')
