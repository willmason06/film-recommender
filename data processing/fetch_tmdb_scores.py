import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import os

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")


ids = np.load("data/movie_ids.npy")

session = requests.Session()

def fetch_scores(tmdb_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

        response = requests.get(
            url,
            params={"api_key": API_KEY},
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        return {
            "id": tmdb_id,
            "release_date": data.get("release_date"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
        }

    except Exception as e:
        print(f"Error for {tmdb_id}: {e}")
        return None


results_list = []

with ThreadPoolExecutor(max_workers=50) as executor:
    results = executor.map(fetch_scores, ids)

    for result in tqdm(results, total=len(ids)):
        if result is not None:
            results_list.append(result)


df = pd.DataFrame(results_list)
df.to_csv("data/movie_scores.csv", index=False)