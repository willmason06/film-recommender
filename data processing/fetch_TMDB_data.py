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


def fetch_movie_data(movie_id):
    session = requests.Session()
    try:
        url = (
            f"{BASE_URL}/movie/{movie_id}"
            f"?api_key={API_KEY}"
            f"&append_to_response=credits,keywords"
        )

        response = session.get(url)
        data = response.json()

        director = next(
            (
                person["name"]
                for person in data["credits"]["crew"]
                if person["job"] == "Director"
            ),
            None
        )

        writers = list(set(
            person["name"]
            for person in data["credits"]["crew"]
            if person["job"] in ["Writer", "Screenplay", "Story"]
        ))

        movie_data = {
            "id": data["id"],
            "title": data["title"],
            "overview": data["overview"],
            "tagline": data["tagline"],
            "genres": [g["name"] for g in data["genres"]],
            "keywords": [k["name"] for k in data["keywords"]["keywords"]],
            "director": director,
            "writers": writers,
            "cast": [actor["name"] for actor in data["credits"]["cast"][:10]],
        }

        return movie_data
    
    except Exception as e:
        print(f"Failed {movie_id}: {e}")
        return None



from concurrent.futures import ThreadPoolExecutor

all_movie_data = []

ids = np.load("data/movie_ids.npy")

with ThreadPoolExecutor(max_workers=50) as executor:
    results = executor.map(fetch_movie_data, ids)

    for movie_data in tqdm(results, total=len(ids)):
        if movie_data is not None:
            all_movie_data.append(movie_data)
        

df = pd.DataFrame(all_movie_data)
df.to_csv("data/tmdb_movie_data.csv", index=False)

print("Saved:", len(df))