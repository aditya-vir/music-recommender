# 🎵 Music Recommender

A content-based music recommendation web app built with Python and Streamlit.

## What it does
Type in a song or artist and get 10 similar track recommendations based on
audio features like danceability, energy, tempo, and more.

## Tech Stack
- Python
- Streamlit
- pandas
- scikit-learn (cosine similarity)
- Plotly

## How to run locally
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) and place `dataset.csv` in the project folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
