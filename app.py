import streamlit as st
import pandas as pd
from recommender import load_data, search_track, get_recommendations, FEATURES
import plotly.express as px

st.set_page_config(page_title="🎵 Music Recommender", layout="centered")
st.title("🎵 Music Recommender")
st.write("Search for a song and get recommendations based on its audio features.")

@st.cache_data
def get_data():
    return load_data()

df = get_data()

query = st.text_input("Search for a song or artist", placeholder="e.g. Blinding Lights")

if query:
    results = search_track(df, query)

    if results.empty:
        st.error("No matching songs found. Try a different search.")
    else:
        options = [f"{row['track_name']} — {row['artists']}" for _, row in results.iterrows()]
        choice = st.selectbox("Select the song you meant:", options)
        chosen_index = results.iloc[options.index(choice)].name

        st.success(f"Finding recommendations for: **{choice}**")

        recs = get_recommendations(df, chosen_index)

        st.subheader("🎧 Recommended Tracks")
        st.dataframe(
            recs[["track_name", "artists", "similarity"]].reset_index(drop=True),
            use_container_width=True
        )

        st.subheader("📊 Audio Feature Comparison")
        seed_row = df.loc[[chosen_index], FEATURES + ["track_name", "artists"]]
        top5 = recs.head(5)
        combined = pd.concat([seed_row, top5[FEATURES + ["track_name", "artists"]]])
        combined["label"] = combined["track_name"] + " - " + combined["artists"]
        melted = combined.melt(id_vars="label", value_vars=FEATURES,
                               var_name="Feature", value_name="Score")
        fig = px.bar(melted, x="Feature", y="Score", color="label",
                     barmode="group", title="Audio Features: Your Song vs Recommendations")
        st.plotly_chart(fig, use_container_width=True)