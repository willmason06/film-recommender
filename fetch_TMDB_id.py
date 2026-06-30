import requests
import pandas as pd
from tqdm import tqdm
import time

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")


BASE_URL = "https://api.themoviedb.org/3"



def fetch_movies(page):
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "sort_by": "vote_average.desc",
        "vote_count.gte": 100,
        "include_adult": False,
        "page": page,
    }
    response = requests.get(url, params=params)
    return response.json()


all_movies = []
TOTAL_PAGES = 500


for page in tqdm(range(1, TOTAL_PAGES + 1)):
    data = fetch_movies(page)

    for movie in data.get("results", []):
        all_movies.append({
            "id": movie["id"],

        })

    time.sleep(0.25)  # avoid rate limits

df = pd.DataFrame(all_movies)
df.to_csv("data/tmdb_movie_ids.csv", index=False)

print("Saved:", len(df))