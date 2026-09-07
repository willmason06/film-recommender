# Film Recommender

A personalised film recommendation pipeline built on top of the TMDB catalogue. It fetches metadata for
~297,000 films, embeds each film as a dense vector, clusters the catalogue to explore structure in the
data, and trains a personal rating-prediction model on a user's own Letterboxd history to surface films
they're likely to enjoy.

## Tech Stack / Features

- **Data collection** — `data processing/fetch_TMDB_id.py` and `fetch_TMDB_data.py` pull film IDs and
  metadata (overview, tagline, genres, keywords, director, writers, top cast) from the TMDB API via
  threaded `requests` calls; `fetch_tmdb_scores.py` separately pulls vote average/count, popularity and
  release date for reranking.
- **Personal data** — `letterboxd_add_ids.py` matches a user's exported Letterboxd ratings/watchlist
  (title + year) to TMDB IDs via the TMDB search endpoint.
- **Embeddings** — `embed_films.py` encodes each film's combined metadata text into a 384-dimensional
  vector using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Dimensionality reduction** — `embedding_reduction.py` reduces the 384D embeddings with PCA (50
  components), then further with UMAP (20 components for clustering, 2 components for visualisation).
- **Clustering** — `cluster.py` runs HDBSCAN on the 20D UMAP embeddings to group similar films;
  `visualise/display_clusters.py` and `visualise/display_umap2.py` inspect cluster membership and plot
  the 2D UMAP projection with matplotlib.
- **Recommendation model** — `test.py` trains a Ridge regression model mapping film embeddings to a
  user's Letterboxd ratings, evaluates it with a train/test split (MAE), then predicts a score for every
  unwatched film and reranks with a metadata-aware formula (predicted rating + vote average, log-scaled
  vote count and popularity, and a recency penalty). `train_model.py` is an earlier draft of the same
  idea using plain linear regression.
- **Similarity lookup** — `main.py` finds the most similar films to a given TMDB ID via cosine similarity
  over the embeddings, and can also blend two films together to find films similar to both.
- **Web app** — `app.py` is a minimal Flask app (`/` and `/recommend/<movie_id>`) with a single template;
  it is currently a scaffold and does not yet call into the trained model.

## How to Run

### 1. Set up the environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure your TMDB API key

Create a `.env` file in the repo root with:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

An API key is free from https://www.themoviedb.org/settings/api.

### 3. Run the data pipeline (in order)

```bash
python "data processing/fetch_TMDB_id.py"       # collect TMDB film IDs
python "data processing/fetch_TMDB_data.py"      # fetch metadata for each ID
python "data processing/fetch_tmdb_scores.py"    # fetch vote/popularity/release data
python "data processing/embed_films.py"          # generate sentence-transformer embeddings
python "data processing/embedding_reduction.py"  # PCA + UMAP reduction
python cluster.py                                # HDBSCAN clustering
```

To bring in your own ratings, export your Letterboxd data (`ratings.csv`, `watchlist.csv`) into
`data/letterboxd/`, then run:

```bash
python "data processing/letterboxd_add_ids.py"
```

### 4. Train the model and get recommendations

```bash
python test.py
```

This trains a Ridge regression model on your Letterboxd ratings, reports MAE on a held-out split, and
prints the top 20 reranked recommendations.

### 5. Explore similarity / clusters

```bash
python main.py                           # find films similar to a given TMDB ID, or blend two films
python visualise/display_clusters.py     # inspect which films fall into a given cluster
python visualise/display_umap2.py        # plot the 2D UMAP projection of the catalogue
```

**Note:** the data-loading paths hardcoded in these scripts (e.g. `data/movie_ids.npy`,
`data/tmdb_movie_data.csv`) assume the pipeline outputs sit directly under `data/`. If your local `data/`
folder uses subfolders (e.g. `data/tmdb/`, `data/embeddings/`), adjust the paths in the corresponding
script, or copy/symlink the files into the flat layout the scripts expect.

## Results

Clustering the full catalogue with HDBSCAN groups the ~297,000-film embedding space into several dozen
genre/theme clusters of varying size (see the cluster size breakdown recorded in
`visualise/display_clusters.py`), alongside a large noise cluster of films that don't fit a tight group.
