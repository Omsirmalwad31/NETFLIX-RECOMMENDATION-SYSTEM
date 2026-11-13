"""
Netflix Recommendation System — Data Preprocessing & Model Building
===================================================================
This script loads, cleans, and merges the two datasets, engineers features,
builds the CountVectorizer + Cosine Similarity model, and serializes the
artifacts for the Streamlit app.
"""

import os
import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# 1. LOAD RAW DATA
# ──────────────────────────────────────────────

print("[1/6] Loading datasets...")
dataset = pd.read_csv("dataset/dataset.csv")
netflix = pd.read_csv("dataset/netflix_titles.csv")

print(f"   dataset.csv  -> {dataset.shape[0]} rows, {dataset.shape[1]} cols")
print(f"   netflix_titles.csv -> {netflix.shape[0]} rows, {netflix.shape[1]} cols")


# ──────────────────────────────────────────────
# 2. CLEAN & PREPARE EACH DATASET
# ──────────────────────────────────────────────

# --- dataset.csv ---
# Keep relevant columns
dataset = dataset[['id', 'title', 'genre', 'overview', 'popularity', 'vote_average']].copy()

# Drop rows where essential columns are missing
dataset.dropna(subset=['title', 'overview'], inplace=True)
dataset.drop_duplicates(subset='title', keep='first', inplace=True)

# Clean genre: "Drama,Crime" -> "Drama Crime"
dataset['genre'] = dataset['genre'].fillna('').apply(lambda x: x.replace(',', ' '))

# Clean overview
dataset['overview'] = dataset['overview'].fillna('')

print(f"\n[2/6] After cleaning dataset.csv: {dataset.shape[0]} rows")


# --- netflix_titles.csv ---
# Keep only Movies (ignore TV Shows for recommendation consistency)
netflix_movies = netflix[netflix['type'] == 'Movie'].copy()

# Keep relevant columns
netflix_movies = netflix_movies[['title', 'director', 'cast', 'listed_in', 'description']].copy()

# Drop duplicates by title
netflix_movies.drop_duplicates(subset='title', keep='first', inplace=True)

# Fill missing values
netflix_movies['director'] = netflix_movies['director'].fillna('')
netflix_movies['cast'] = netflix_movies['cast'].fillna('')
netflix_movies['listed_in'] = netflix_movies['listed_in'].fillna('')
netflix_movies['description'] = netflix_movies['description'].fillna('')

# Clean cast: keep only first 3 actors to avoid noise
def clean_cast(cast_str):
    """Take the first 3 cast members and remove spaces in names."""
    if not cast_str:
        return ''
    actors = [a.strip().replace(' ', '') for a in cast_str.split(',')[:3]]
    return ' '.join(actors)

netflix_movies['cast'] = netflix_movies['cast'].apply(clean_cast)

# Clean director: remove spaces in name
netflix_movies['director'] = netflix_movies['director'].apply(
    lambda x: x.strip().replace(' ', '') if x else ''
)

# Clean listed_in (Netflix genres): "Documentaries, Dramas" -> "Documentaries Dramas"
netflix_movies['listed_in'] = netflix_movies['listed_in'].apply(
    lambda x: x.replace(',', ' ').strip()
)

print(f"   After cleaning netflix_titles.csv: {netflix_movies.shape[0]} movie rows")


# ──────────────────────────────────────────────
# 3. MERGE DATASETS
# ──────────────────────────────────────────────

# Normalize titles for matching
dataset['title_lower'] = dataset['title'].str.strip().str.lower()
netflix_movies['title_lower'] = netflix_movies['title'].str.strip().str.lower()

# Left-join: keep all movies from dataset.csv, enrich with netflix cast/director
merged = dataset.merge(
    netflix_movies[['title_lower', 'director', 'cast']],
    on='title_lower',
    how='left'
)

# Fill any NaN introduced by the merge
merged['director'] = merged['director'].fillna('')
merged['cast'] = merged['cast'].fillna('')

# Drop helper column
merged.drop(columns=['title_lower'], inplace=True)

# Reset index
merged.reset_index(drop=True, inplace=True)

print(f"\n[3/6] Merged dataset: {merged.shape[0]} rows")
print(f"   Columns: {list(merged.columns)}")


# ──────────────────────────────────────────────
# 4. FEATURE ENGINEERING — BUILD TAGS
# ──────────────────────────────────────────────

def build_tags(row):
    """Combine genre, overview, cast, and director into a single text blob."""
    parts = [
        str(row['genre']),
        str(row['overview']),
        str(row['cast']),
        str(row['director']),
    ]
    return ' '.join(parts).lower().strip()

merged['tags'] = merged.apply(build_tags, axis=1)

print(f"\n[4/6] Sample tags (first movie):")
print(f"   {merged['tags'].iloc[0][:200]}...")


# ──────────────────────────────────────────────
# 5. VECTORIZATION & COSINE SIMILARITY
# ──────────────────────────────────────────────

print("\n[5/6] Vectorizing with CountVectorizer (max_features=5000)...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(merged['tags']).toarray()

print(f"   Vector matrix shape: {vectors.shape}")

print("   Computing cosine similarity matrix...")
similarity = cosine_similarity(vectors)
print(f"   Similarity matrix shape: {similarity.shape}")


# ──────────────────────────────────────────────
# 6. SAVE MODEL ARTIFACTS
# ──────────────────────────────────────────────

os.makedirs('model', exist_ok=True)

# Save the movie dataframe (only columns needed by the app)
movies_df = merged[['id', 'title', 'genre', 'overview', 'popularity', 'vote_average']].copy()
movies_dict = movies_df.to_dict()

with open('model/movies_dict.pkl', 'wb') as f:
    pickle.dump(movies_dict, f)
    print(f"\n[6/6] Saved model/movies_dict.pkl ({movies_df.shape[0]} movies)")

with open('model/similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)
    print(f"   Saved model/similarity.pkl ({similarity.shape})")

print("\n=== Preprocessing complete! You can now run: streamlit run app.py ===")
