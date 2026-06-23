# Implementation Plan - Movie Recommendation System

Setup all the requirements and build a high-quality Movie Recommendation System inside the Jupyter Notebook `Rec_sys.ipynb` using the MovieLens 32M dataset.

## User Review Required

> [!IMPORTANT]
> **Dataset Size & Performance Constraints**: The `ratings.csv` file is 877 MB and contains 32 million records. Loading and computing similarities on the entire dataset will consume significant memory (potentially exceeding 8-16 GB of RAM) and CPU time.
> - **Our Approach**: We will implement a robust filtering strategy to select a sub-sample of the dataset (e.g., top $N$ most active users and $M$ most rated movies) or take a representative subset of users (e.g. 5,000-10,000 users and all their ratings) for building the Collaborative Filtering models. This ensures the notebook runs quickly and smoothly on standard laptops.
> - **Dependency Installation**: `scikit-surprise` is a popular library for recommender systems, but compiling it on Windows can sometimes fail if visual C++ build tools are missing. To guarantee a frictionless setup, we will implement Matrix Factorization (SVD) and Collaborative Filtering using standard, pre-compiled scientific libraries (`scikit-learn`, `scipy`, `numpy`, and `pandas`). We will still attempt to install `scikit-surprise` in the requirements, but we will make sure our code uses pure scipy/scikit-learn/pandas implementations so it runs regardless.

## Proposed Changes

### Requirements Setup

#### [NEW] [requirements.txt](file:///C:/Users/VISHWAJEET/OneDrive/Recommendatin_System/requirements.txt)
We will create a `requirements.txt` file listing all the essential packages needed for data analysis, visualization, and building the recommendation models.
The requirements will include:
- `pandas`
- `numpy`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `ipykernel` (to run the Jupyter notebook)

### Notebook Structure

#### [MODIFY] [Rec_sys.ipynb](file:///C:/Users/VISHWAJEET/OneDrive/Recommendatin_System/Rec_sys.ipynb)
We will populate the notebook with a clean, well-commented, and premium-looking structure:

1. **Step 1: Installation & Setup**
   - Pip installation commands for requirements.
   - Importing libraries.
   - Setting up paths to `ml-32m` csv files.
2. **Step 2: Exploratory Data Analysis (EDA)**
   - Load movies, ratings, and tags.
   - Visualizing distribution of ratings (using seaborn/matplotlib).
   - Analysis of most popular movies (by rating count).
   - Analysis of highest-rated movies (using a weighted average rating formula to avoid bias from single ratings).
   - Genre popularity analysis.
3. **Step 3: Data Preprocessing & Sampling**
   - Subsampling the 32M ratings to keep training times fast (e.g., users with $\ge 150$ ratings and movies with $\ge 500$ ratings, or a random sample of users and all their ratings).
   - Creating train-test splits.
4. **Step 4: Recommender System Models**
   - **Model A: Popularity-Based / Demographic Recommender**
     - Global recommendations using IMDB's weighted rating formula.
   - **Model B: Content-Based Recommender**
     - Using movie titles, genres, and user tags.
     - Building a TF-IDF matrix of movie metadata.
     - Computing Cosine Similarity between movies.
     - Recommending movies similar to a given movie title.
   - **Model C: Collaborative Filtering (Matrix Factorization using SVD)**
     - Building a user-movie interaction matrix (sparse format).
     - Running Singular Value Decomposition (SVD) using `scipy.sparse.linalg.svds` or `scikit-learn.decomposition.TruncatedSVD`.
     - Predicting missing ratings.
5. **Step 5: Model Evaluation**
   - Calculating RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error) for the Collaborative Filtering predictions on the test set.
6. **Step 6: Interactive Recommendations**
   - Function `recommend_movies_for_user(user_id, top_n)`: Shows a user's top-rated movies and then lists recommended movies with predicted ratings.
   - Function `recommend_similar_movies(movie_title, top_n)`: Content-based recommendation of similar movies.

## Verification Plan

### Automated Verification
We will run a script to install the requirements and verify that:
- Python environment can import all dependencies without error.
- The Jupyter notebook can run cells successfully.

### Manual Verification
- We can inspect the notebook output and plots in the IDE or run a command to run the notebook and check that it completes.
