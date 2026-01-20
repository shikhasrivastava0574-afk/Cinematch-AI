import streamlit as st
import joblib
import pandas as pd
import requests
import os

# ================== PATH SETUP ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== CONFIG ==================
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

TMDB_API_KEY = "a79fe81a457c7863c127f3d2a2caca8e"

# ================== DARK THEME ==================
st.markdown("""
<style>
.stApp { background-color:#0f172a; color:white; }
h1,h2,h3,h4,h5,h6,p,span,div { color:white !important; }
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODELS ==================
user_item_matrix = joblib.load(os.path.join(BASE_DIR, "models", "user_item_matrix.pkl"))
user_similarity_df = joblib.load(os.path.join(BASE_DIR, "models", "user_similarity.pkl"))

# ================== LOAD HOLLYWOOD DATA (MovieLens) ==================
genre_cols = [
    "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
    "Romance","Sci-Fi","Thriller","War","Western"
]

movies = pd.read_csv(
    os.path.join(BASE_DIR, "data", "raw", "u.item"),
    sep="|",
    encoding="latin-1",
    header=None
)

movies = movies.iloc[:, :2+19]
movies.columns = ["content_id", "title"] + genre_cols

# ================== LOAD TRUE BOLLYWOOD DATA ==================
bollywood_movies = pd.read_csv(
    os.path.join(BASE_DIR, "data", "bollywood_movies_imdb.csv")
)
bollywood_movies["primaryTitle"] = bollywood_movies["primaryTitle"].str.strip()

# ================== SIDEBAR ==================
st.sidebar.title("🎯 Recommendation Settings")

# Choose Industry
selected_industry = st.sidebar.selectbox(
    "🎥 Choose Industry",
    ["Hollywood", "Bollywood"]
)

# Number of recommendations
num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    1, 10, 5
)

# Choose Genre
selected_genre = st.sidebar.selectbox(
    "🎭 Choose Genre",
    genre_cols[1:]
)

# User ID only for Hollywood
if selected_industry == "Hollywood":
    user_id = st.sidebar.number_input(
        "👤 Enter User ID (Hollywood only)",
        min_value=int(user_item_matrix.index.min()),
        max_value=int(user_item_matrix.index.max()),
        step=1
    )
else:
    user_id = None
    st.sidebar.info("🔒 User ID is not required for Bollywood recommendations")

# ================== HEADER ==================
# ================== HEADER ==================
st.markdown("""
<h1 style='text-align:center; color:#38bdf8;'>🎬 CineMatch AI</h1>
<p style='text-align:center; font-size:18px; color:#cbd5f5;'>
A Hybrid AI-Powered Movie Recommendation Engine
</p>
<p style='text-align:center; font-size:15px; color:#94a3b8;'>
Hollywood & Bollywood | Personalized | Smart | Fast
</p>
""", unsafe_allow_html=True)

st.divider()


# ================== TMDB POSTER FUNCTION ==================
def get_movie_poster(title):
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data.get("results"):
            poster = data["results"][0].get("poster_path")
            if poster:
                return f"https://image.tmdb.org/t/p/w500{poster}"
    except:
        pass
    return None

# ================== HYBRID RECOMMENDER ==================
def recommend(user_id, n=5, genre=None, industry=None):

    # ---------- BOLLYWOOD (CONTENT-BASED) ----------
    if industry == "Bollywood":
        bolly = bollywood_movies.copy()

        if genre:
            bolly = bolly[bolly["genres"].str.contains(genre, case=False, na=False)]

        if len(bolly) == 0:
            return []

        sample = bolly.sample(n=min(n, len(bolly)))
        return [(row["primaryTitle"], "IMDb") for _, row in sample.iterrows()]

    # ---------- HOLLYWOOD (COLLABORATIVE FILTERING + GENRE FIRST) ----------
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    watched = user_item_matrix.loc[user_id]
    watched = watched[watched > 0].index.tolist()

    # Step 1: restrict candidate movies by genre
    if genre:
        genre_movie_ids = movies[movies[genre] == 1]["content_id"].tolist()
    else:
        genre_movie_ids = movies["content_id"].tolist()

    scores = {}

    # Step 2: score only genre-matching movies
    for sim_user, similarity in similar_users.items():
        user_ratings = user_item_matrix.loc[sim_user]
        for movie_id, rating in user_ratings.items():
            if (
                movie_id not in watched and
                movie_id in genre_movie_ids and
                rating > 0
            ):
                scores[movie_id] = scores.get(movie_id, 0) + rating * similarity

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    final_recs = []
    for movie_id, score in ranked:
        movie_row = movies[movies["content_id"] == movie_id]
        if movie_row.empty:
            continue
        title = movie_row["title"].values[0]
        final_recs.append((title, round(score, 2)))
        if len(final_recs) == n:
            break

    # Step 3: fallback if genre is too strict
    if len(final_recs) == 0:
        scores = {}
        for sim_user, similarity in similar_users.items():
            user_ratings = user_item_matrix.loc[sim_user]
            for movie_id, rating in user_ratings.items():
                if movie_id not in watched and rating > 0:
                    scores[movie_id] = scores.get(movie_id, 0) + rating * similarity

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for movie_id, score in ranked:
            movie_row = movies[movies["content_id"] == movie_id]
            if movie_row.empty:
                continue
            title = movie_row["title"].values[0]
            final_recs.append((title, round(score, 2)))
            if len(final_recs) == n:
                break

    return final_recs


# ================== BUTTON ==================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_btn = st.button("✨ Get Recommendations")

# ================== RESULTS ==================
if recommend_btn:
    with st.spinner("Finding movies for you... 🍿"):
        recs = recommend(user_id, num_recommendations, selected_genre, selected_industry)

    if not recs:
        st.warning("No movies found. Try changing genre or industry.")
    else:
        st.success(f"🎯 {selected_industry} | {selected_genre} Movies Recommended")

        for i, (movie, score) in enumerate(recs, start=1):
            poster = get_movie_poster(movie)

            col1, col2 = st.columns([1, 3])
            with col1:
                if poster:
                    st.image(poster, width=130)
                else:
                    st.image("https://via.placeholder.com/130x200.png?text=No+Poster", width=130)

            with col2:
                st.markdown(
                    f"""
                    <div style="
                        background-color:#1e293b;
                        padding:16px;
                        border-radius:12px;
                        margin-bottom:14px;
                        box-shadow:0px 4px 12px rgba(0,0,0,0.4);
                        font-size:18px;
                    ">
                    <b style="color:#facc15;">#{i}</b> 🎬 <b>{movie}</b><br>
                    🎥 Industry: {selected_industry}<br>
                    🎭 Genre: {selected_genre}<br>
                    ⭐ Score: {score}<br>
                    🍿 Perfect match for your taste!
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.divider()
st.markdown(
    "<p style='text-align:center; color:#94a3b8;'>Built with ❤️ using Python, ML, IMDb & Streamlit</p>",
    unsafe_allow_html=True
)
