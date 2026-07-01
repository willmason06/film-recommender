import requests
import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"

def find_tmdb_id(title, year):
    url = f"{BASE_URL}/search/movie"

    params = {
        "api_key": API_KEY,
        "query": title,
        "year": year
    }

    response = requests.get(url, params=params)
    results = response.json()["results"]

    if len(results) == 0:
        return None

    return results[0]["id"]

file = 'data/letterboxd/ratings.csv'