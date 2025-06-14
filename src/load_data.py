import nltk
from nltk.corpus import movie_reviews
import pandas as pd

def load_movie_review_data():
    nltk.download("movie_reviews")
    documents = [
        (" ".join(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    df = pd.DataFrame(documents, columns=["review", "sentiment"])
    return df
