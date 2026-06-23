import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os

# Set page config for a premium wide layout
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium dark-mode aesthetics (Glassmorphism & animations)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Background and text color adjustments */
    .main {
        background-color: #0d0f14;
    }
    
    /* Header Gradient Text */
    .title-text {
        background: linear-gradient(135deg, #ff007f, #7f00ff, #00f0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .subtitle-text {
        color: #8f9cae;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Movie Card Styling (Glassmorphism) */
    .movie-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    /* Glow effect on hover */
    .movie-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 40px 0 rgba(127, 0, 255, 0.15);
    }
    
    /* Card Title */
    .movie-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    
    /* Genre tags container */
    .genre-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 0.8rem;
    }
    
    /* Genre Pill Badge */
    .genre-badge {
        background: rgba(127, 0, 255, 0.15);
        color: #b280ff;
        border: 1px solid rgba(127, 0, 255, 0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Ratings area */
    .score-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 0.8rem;
    }
    
    .rating-value {
        color: #ffd700;
        font-weight: 600;
        font-size: 1.1rem;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .similarity-value {
        color: #00f0ff;
        font-weight: 600;
        font-size: 0.9rem;
        background: rgba(0, 240, 255, 0.1);
        border: 1px solid rgba(0, 240, 255, 0.2);
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* Liked past movie cards */
    .liked-card {
        background: rgba(255, 0, 127, 0.03);
        border-left: 4px solid #ff007f;
    }
    
</style>
""", unsafe_allow_html=True)


# --- Load Models & Data (with Cache for performance) ---
@st.cache_resource
def load_all_assets():
    if not (os.path.exists('svd_recommender.joblib') and 
            os.path.exists('tfidf_vectorizer.joblib') and 
            os.path.exists('tfidf_matrix.joblib') and 
            os.path.exists('movies_catalog.joblib')):
        return None
        
    svd = joblib.load('svd_recommender.joblib')
    tfidf_vec = joblib.load('tfidf_vectorizer.joblib')
    tfidf_mat = joblib.load('tfidf_matrix.joblib')
    movies_df = joblib.load('movies_catalog.joblib')
    
    return svd, tfidf_vec, tfidf_mat, movies_df

assets = load_all_assets()

if assets is None:
    st.error(" Model files not found! Please run the export cells in your Jupyter notebook `Rec_sys.ipynb` first to generate the `.joblib` files.")
    st.stop()

svd, tfidf_vec, tfidf_mat, movies_df = assets

# Unpack SVD variables
U = svd['U']
sigma_diag = svd['sigma_diag']
Vt = svd['Vt']
user_means = svd['user_means']
user_id_to_code = svd['user_id_to_code']
movie_id_to_code = svd['movie_id_to_code']
code_to_movie_id = svd['code_to_movie_id']
num_movies = len(movie_id_to_code)

# Global variables for rating mean calculation fallback
C = 3.0 # Fallback mean rating

# --- Helper Functions ---
def render_stars(rating_score):
    stars = int(round(rating_score))
    return "★" * stars + "☆" * (5 - stars)

# --- Header Section ---
st.markdown('<div class="title-text">RECOMMENDO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">A hybrid movie recommendation platform leveraging Content-Based TF-IDF & Collaborative SVD Matrix Factorization</div>', unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.markdown("### 🎬 Recommendation settings")

# User Selection (for SVD personalization)
available_users = list(user_id_to_code.keys())
selected_user_id = st.sidebar.selectbox(
    "Select User ID Profile:",
    options=available_users,
    index=0
)

# Number of suggestions slider
top_n = st.sidebar.slider(
    "Number of recommendations:",
    min_value=3,
    max_value=12,
    value=6,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How this works:**
1. **User Profile**: Personalizes results using Singular Value Decomposition (SVD) on user-rating behaviors.
2. **Search Similar**: Suggests content-similar movies using TF-IDF text features & Cosine Similarity.
3. **IMDb Charts**: Fallback ranking utilizing IMDb weighted rating formula.
""")

# --- Tab Layout ---
tab_personal, tab_similar, tab_charts = st.tabs([
    "🎯 Personalized for You", 
    "🔍 Similar Movie Search", 
    "🔥 IMDb Top Charts"
])

# ==========================================
# TAB 1: PERSONALIZED RECOMMENDATIONS (SVD)
# ==========================================
with tab_personal:
    st.subheader(f"Recommendations for User {selected_user_id}")
    
    # 1. Load User's Past Favorites
    # Get user code
    user_code = user_id_to_code[selected_user_id]
    
    # Predict rating vector
    user_U = U[user_code, :]
    user_pred = np.dot(np.dot(user_U, sigma_diag), Vt) + user_means[user_code]
    
    # Display user's top liked movies from the raw dataset as a reference context
    # Note: Since raw ratings might not be fully loaded here, we can infer what they liked or load the joblib rating stats
    # We will display movies with high predicted scores that are part of the catalog, but to show actual user likes,
    # we can show the top 3 highest predicted ratings that they have rated (which represents SVD modeling fit).
    # Even better, we can show their top recommendations in a nice grid.
    
    # Predict ratings for all movies in catalog
    movies_copy = movies_df.copy()
    
    # Map movieId to code
    movies_copy['code'] = movies_copy['movieId'].map(movie_id_to_code)
    # Filter catalog to only movies in SVD subset
    movies_subset = movies_copy.dropna(subset=['code']).copy()
    movies_subset['code'] = movies_subset['code'].astype(int)
    
    # Add SVD predictions to subset
    movies_subset['svd_prediction'] = movies_subset['code'].map(lambda c: user_pred[c])
    movies_subset['svd_prediction'] = np.clip(movies_subset['svd_prediction'], 0.5, 5.0)
    
    # Show Top Recommendations (highest SVD predictions)
    # We sort by predictions descending
    recommendations = movies_subset.sort_values(by='svd_prediction', ascending=False).head(top_n)
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown(f"#### 👤 User Info")
        st.metric(label="User Code", value=f"#{user_code}")
        st.metric(label="Base Mean Rating", value=f"{user_means[user_code]:.2f} / 5.0")
        
    with col2:
        st.markdown(f"#### 🚀 Top Recommendations for User {selected_user_id}")
        
        # Render cards in a grid layout (3 per row)
        cols_per_row = 3
        for i in range(0, len(recommendations), cols_per_row):
            row_movies = recommendations.iloc[i:i+cols_per_row]
            grid_cols = st.columns(cols_per_row)
            
            for idx, (_, movie) in enumerate(row_movies.iterrows()):
                genres = movie['genres'].split('|')
                genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres[:3]])
                
                with grid_cols[idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="genre-container">{genre_html}</div>
                        <div class="score-container">
                            <div class="rating-value">⭐ {movie['svd_prediction']:.2f}</div>
                            <div class="similarity-value">Predicted Score</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 2: SEARCH SIMILAR MOVIES (HYBRID)
# ==========================================
with tab_similar:
    st.subheader("Search Similar Movies")
    st.markdown("Search for any movie to get personalized content-based suggestions re-ranked by your user profile.")
    
    # Autocomplete select box
    all_titles = sorted(movies_df['title'].tolist())
    search_movie = st.selectbox(
        "Type or Select a movie:",
        options=all_titles,
        index=0
    )
    
    if search_movie:
        # Get target movie details
        target_movie = movies_df[movies_df['title'] == search_movie].iloc[0]
        target_idx = movies_df[movies_df['title'] == search_movie].index[0]
        
        st.markdown(f"**Selected Seed Movie:** *{target_movie['title']}* | **Genre:** {target_movie['genres']}")
        
        # 1. Content-based: Cosine Similarity on TF-IDF Matrix
        sim_scores = cosine_similarity(tfidf_mat[target_idx], tfidf_mat).flatten()
        # Get top 30 similar indices (exclude the movie itself)
        similar_indices = sim_scores.argsort()[-31:-1][::-1]
        
        similar_movies = movies_df.iloc[similar_indices].copy()
        similar_movies['similarity_score'] = sim_scores[similar_indices]
        
        # 2. Personalization: Re-rank by SVD prediction for the selected user profile
        user_code = user_id_to_code[selected_user_id]
        user_U = U[user_code, :]
        
        predictions = []
        for _, row in similar_movies.iterrows():
            m_id = row['movieId']
            if m_id in movie_id_to_code:
                code = movie_id_to_code[m_id]
                pred = np.dot(np.dot(user_U, sigma_diag), Vt[:, code]) + user_means[user_code]
                predictions.append(np.clip(pred, 0.5, 5.0))
            else:
                # Fallback to movie's weighted rating if movie is not in SVD subset
                predictions.append(row['weighted_rating'])
                
        similar_movies['personalized_rating'] = predictions
        
        # Sort by SVD predicted rating
        hybrid_recommendations = similar_movies.sort_values(by='personalized_rating', ascending=False).head(top_n)
        
        # Render cards
        cols_per_row = 3
        for i in range(0, len(hybrid_recommendations), cols_per_row):
            row_movies = hybrid_recommendations.iloc[i:i+cols_per_row]
            grid_cols = st.columns(cols_per_row)
            
            for idx, (_, movie) in enumerate(row_movies.iterrows()):
                genres = movie['genres'].split('|')
                genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres[:3]])
                match_percentage = int(movie['similarity_score'] * 100)
                
                with grid_cols[idx]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="genre-container">{genre_html}</div>
                        <div class="score-container">
                            <div class="rating-value">⭐ {movie['personalized_rating']:.2f}</div>
                            <div class="similarity-value">{match_percentage}% Match</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# TAB 3: IMDB TOP CHARTS (DEMOGRAPHIC)
# ==========================================
with tab_charts:
    st.subheader("Global Popularity Top Charts")
    st.markdown("These are the highest globally rated movies using the IMDb Weighted Ratings formula (accounting for rating volume).")
    
    # Sort catalog by weighted_rating
    top_charts = movies_df.sort_values(by='weighted_rating', ascending=False).head(top_n)
    
    # Render cards
    cols_per_row = 3
    for i in range(0, len(top_charts), cols_per_row):
        row_movies = top_charts.iloc[i:i+cols_per_row]
        grid_cols = st.columns(cols_per_row)
        
        for idx, (_, movie) in enumerate(row_movies.iterrows()):
            genres = movie['genres'].split('|')
            genre_html = "".join([f'<span class="genre-badge">{g}</span>' for g in genres[:3]])
            
            with grid_cols[idx]:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{movie['title']}</div>
                    <div class="genre-container">{genre_html}</div>
                    <div class="score-container">
                        <div class="rating-value">⭐ {movie['weighted_rating']:.2f}</div>
                        <div class="similarity-value">IMDb Weighted</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
