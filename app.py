

import os
import pickle
import requests
import streamlit as st
import pandas as pd


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────
# CUSTOM CSS — NETFLIX-INSPIRED DARK THEME
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a0a 50%, #0a0a0a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Hide Streamlit defaults ── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Hero Section ── */
    .hero-container {
        text-align: center;
        padding: 2rem 1rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E50914 0%, #FF6B6B 50%, #E50914 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s ease-in-out infinite;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 200% center; }
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #888;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── Divider ── */
    .red-divider {
        width: 80px;
        height: 3px;
        background: linear-gradient(90deg, #E50914, #FF6B6B);
        border-radius: 2px;
        margin: 1rem auto 2rem auto;
    }

    /* ── Search Section ── */
    .search-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(229,9,20,0.15);
        border-radius: 16px;
        padding: 2rem;
        margin: 0 auto 2.5rem auto;
        max-width: 700px;
        backdrop-filter: blur(10px);
    }
    .search-label {
        font-size: 1rem;
        color: #ccc;
        font-weight: 500;
        margin-bottom: 0.8rem;
        text-align: center;
    }

    /* ── Selectbox styling ── */
    div[data-baseweb="select"] > div {
        background-color: rgba(30,30,30,0.9) !important;
        border: 1px solid rgba(229,9,20,0.3) !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #E50914 !important;
    }

    /* ── Button styling ── */
    .stButton > button {
        background: linear-gradient(135deg, #E50914 0%, #B20710 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(229,9,20,0.3) !important;
        width: 100%;
        margin-top: 0.5rem;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(229,9,20,0.5) !important;
        background: linear-gradient(135deg, #FF1A25 0%, #E50914 100%) !important;
    }

    /* ── Results Header ── */
    .results-header {
        text-align: center;
        margin: 2rem 0 1.5rem 0;
    }
    .results-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
    }
    .results-title span {
        color: #E50914;
    }

    /* ── Movie Card ── */
    .movie-card {
        background: rgba(20,20,20,0.8);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-8px) scale(1.02);
        border-color: rgba(229,9,20,0.4);
        box-shadow: 0 20px 40px rgba(229,9,20,0.15), 0 8px 16px rgba(0,0,0,0.4);
    }
    .movie-poster {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
    }
    .movie-info {
        padding: 1rem;
    }
    .movie-title-card {
        font-size: 0.95rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.4rem;
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .movie-genre {
        font-size: 0.75rem;
        color: #E50914;
        font-weight: 500;
        margin-bottom: 0.4rem;
    }
    .movie-rating {
        font-size: 0.8rem;
        color: #ffd700;
    }

    /* ── Stats Bar ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1.5rem;
        margin: 0 auto 2rem auto;
        max-width: 600px;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E50914;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.2rem;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 2rem 1rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
    .footer p {
        color: #444;
        font-size: 0.8rem;
    }
    .footer span {
        color: #E50914;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .hero-title { font-size: 2rem; }
        .stats-bar { gap: 1.5rem; }
        .stat-number { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD MODEL ARTIFACTS
# ──────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load the preprocessed movie data and similarity matrix."""
    movies_dict = pickle.load(open('model/movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('model/similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()


# ──────────────────────────────────────────────
# POSTER FETCHING
# ──────────────────────────────────────────────

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
PLACEHOLDER_POSTER = "https://via.placeholder.com/500x750/141414/E50914?text=No+Poster"

@st.cache_data(ttl=3600)
def fetch_poster(movie_id):
    """Fetch movie poster URL from TMDB API."""
    if not TMDB_API_KEY:
        return PLACEHOLDER_POSTER
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=5)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return PLACEHOLDER_POSTER


# ──────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ──────────────────────────────────────────────

def recommend(movie_title):
    """Find the top 5 most similar movies using cosine similarity."""
    movie_index = movies[movies['title'] == movie_title].index[0]
    distances = similarity[movie_index]
    # Sort by similarity (descending), skip the first one (itself)
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended = []
    for idx, score in movies_list:
        row = movies.iloc[idx]
        recommended.append({
            'title': row['title'],
            'id': row['id'],
            'genre': row['genre'],
            'vote_average': row['vote_average'],
            'overview': row['overview'],
            'similarity': round(score * 100, 1),
        })
    return recommended


# ──────────────────────────────────────────────
# UI — HERO SECTION
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
    <div class="hero-title">NETFLIX RECOMMENDATIONS</div>
    <div class="hero-subtitle">Content-Based Movie Recommendation System</div>
    <div class="red-divider"></div>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown(f"""
<div class="stats-bar">
    <div class="stat-item">
        <div class="stat-number">{len(movies):,}</div>
        <div class="stat-label">Movies</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">5</div>
        <div class="stat-label">Recommendations</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">ML</div>
        <div class="stat-label">Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI — SEARCH SECTION
# ──────────────────────────────────────────────

st.markdown('<div class="search-section">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Select a movie you like</div>', unsafe_allow_html=True)

selected_movie = st.selectbox(
    label="Movie",
    options=movies['title'].values,
    index=0,
    label_visibility="collapsed",
    key="movie_select",
)

recommend_clicked = st.button("Get Recommendations", key="recommend_btn")
st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# UI — RECOMMENDATIONS DISPLAY
# ──────────────────────────────────────────────

if recommend_clicked:
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(selected_movie)

    # Results header
    st.markdown(f"""
    <div class="results-header">
        <div class="results-title">
            Because you liked <span>"{selected_movie}"</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Display 5 movie cards in columns
    cols = st.columns(5, gap="medium")

    for i, (col, movie) in enumerate(zip(cols, recommendations)):
        with col:
            poster_url = fetch_poster(movie['id'])
            rating_stars = "★" * int(round(movie['vote_average'] / 2))

            st.markdown(f"""
            <div class="movie-card">
                <img class="movie-poster" src="{poster_url}" alt="{movie['title']}"
                     onerror="this.src='{PLACEHOLDER_POSTER}'"/>
                <div class="movie-info">
                    <div class="movie-title-card">{movie['title']}</div>
                    <div class="movie-genre">{movie['genre']}</div>
                    <div class="movie-rating">{rating_stars} {movie['vote_average']}/10</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

