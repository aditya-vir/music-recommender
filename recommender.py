import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["danceability", "energy", "speechiness",
            "acousticness", "instrumentalness", "valence", "tempo"]

def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.dropna(subset=FEATURES + ["track_name", "artists"])
    df = df.drop_duplicates(subset=["track_name", "artists"])
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df

def search_track(df, query):
    query = query.lower()
    match = df[
        df["track_name"].str.lower().str.contains(query) |
        df["artists"].str.lower().str.contains(query)
    ]
    return match.head(5)  # return top 5 matches for user to pick from

def get_recommendations(df, track_index, n=10):
    seed = df.loc[[track_index], FEATURES]
    all_features = df[FEATURES]
    similarities = cosine_similarity(seed, all_features)[0]
    df = df.copy()
    df["similarity"] = similarities
    recs = df[df.index != track_index].sort_values("similarity", ascending=False).head(n)
    return recs[["track_name", "artists", "similarity"] + FEATURES]