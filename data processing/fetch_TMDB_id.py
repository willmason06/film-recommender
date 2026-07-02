import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from dotenv import load_dotenv

import os

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

existing_ids = set(np.load("data/movie_ids.npy"))

def fetch_movies(page, sort_method, year):
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "sort_by": sort_method,
        "vote_count.gte": 2,
        "primary_release_year": year,
        "include_adult": False,
        "page": page,
    }
    response = requests.get(url, params=params)
    return response.json()


sort_methods = ["vote_average.desc", "popularity.desc", "revenue.desc", "vote_count.desc", "primary_release_date.desc"]

new_ids_added = 0
tasks = []

for year in range(1900, 2027):
    for sort_method in tqdm(sort_methods):
        first_page = fetch_movies(1, sort_method, year)
        total_pages = min(first_page["total_pages"], 500)

        for page in tqdm(range(1, total_pages + 1)):

              tasks.append(
                (page, sort_method, year)
            )


from concurrent.futures import ThreadPoolExecutor

def fetch_task(task):
    page, sort_method, year = task

    try:
        return fetch_movies(page, sort_method, year)
    except Exception:
        return None

with ThreadPoolExecutor(max_workers=20) as executor:
    results = executor.map(fetch_task, tasks)

    for data in tqdm(results, total=len(tasks)):

        if data is None:
            continue

        for movie in data.get("results", []):
            movie_id = movie["id"]

            if movie_id in existing_ids:
                continue

            existing_ids.add(movie_id)
            new_ids_added += 1

            if new_ids_added % 1000 == 0:
                np.save(
                    "data/movie_ids.npy",
                    np.array(list(existing_ids))
                )

                print(
                    f"Added {new_ids_added} new IDs "
                    f"({len(existing_ids)} total)"
                )



np.save(
    "data/movie_ids.npy",
    np.array(list(existing_ids))
)

# add any missing letterboxd ids
ids_set = set(np.load("data/movie_ids.npy"))
df_letterboxd_ratings = pd.read_csv('data/letterboxd/ratings_withid.csv')
df_letterboxd_watchlist = pd.read_csv('data/letterboxd/watchlist_withid.csv')

watched_ids = df_letterboxd_ratings['id']
watchlist_ids = df_letterboxd_watchlist['id']

ids_set.update(watched_ids)
ids_set.update(watchlist_ids)

np.save(
    "data/movie_ids.npy",
    np.array(list(ids_set))
)

