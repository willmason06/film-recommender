'''
want to fetch:
/movie/{id}
/movie/{id}/credits
/movie/{id}/keywords


add function to stop it being done if id is already in output file
'''

import requests
import pandas as pd
from tqdm import tqdm
import time

from dotenv import load_dotenv
import os
load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"


def fetch_movie_data(movie_id):
    url = (
        f"{BASE_URL}/movie/{movie_id}"
        f"?api_key={API_KEY}"
        f"&append_to_response=credits,keywords"
    )

    response = requests.get(url)
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



all_movie_data = []

id_file = "data/tmdb_movie_ids.csv"

df = pd.read_csv(id_file)

for movie_id in tqdm(df['id'][:10]):
    all_movie_data.append(fetch_movie_data(movie_id))

    time.sleep(0.25)  # avoid rate limits


df = pd.DataFrame(all_movie_data)
df.to_csv("data/tmdb_movie_data.csv", index=False)

print("Saved:", len(df))