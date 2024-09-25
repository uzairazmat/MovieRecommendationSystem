import streamlit as st
import  pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}"
    "?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US)".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

with open("movies.pkl", "rb") as file:
    movies = pickle.load(file)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000 , stop_words="english")
vector = cv.fit_transform(movies["tags"]).toarray()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)


# RECOMMENDER FUNCTION

def recommend(title):
    # Use the variable title instead of the string "title"
    matching_movies = movies[movies["title"] == title]

    # Check if any matching movie was found
    if matching_movies.empty:
        return "Movie not found"

    index = matching_movies.index[0]  # Get the first matching index

    distances = similarity[index]  # Access similarity scores
    # Get the top 5 similar movies (excluding the first one which is the same movie)
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    # Create a list of recommended movie titles
    recommended_movies = []
    recommended_poster = []
    for i in movies_list:
        movies_id = movies.iloc[i[0]].movie_id
        #fectch from API
        recommended_movies.append(movies.iloc[i[0]]['title'])
        recommended_poster.append(fetch_poster(movies_id))

    return recommended_movies,recommended_poster


st.title("Movie Recommendation system")

selected_movie_name = st.selectbox(
"Get your personalized movie picks",
movies["title"],
)
if st.button("Recommend"):
    names,posters = recommend(selected_movie_name)

    col1, col2, col3 ,col4 , col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
st.divider()
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")
st.text("")









st.header("How to use?")
st.video("Screencast from 25-09-2024 12:22:57.webm")




