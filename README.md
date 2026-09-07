# film-recommender

Uses TMDB API to gather metadata on almost 300k of its most popular films. 
Then a sentence transformer maps metadata to 384D vectors. 
Vectors are clustered to group similar films together. 
Personal data is imported from letterboxdand and using my ratings, a ML model is trained.
Model can be used to predict ratings on unwatched films. 

Also multiple films can be inputted and itll find films that are a blend of the two.
