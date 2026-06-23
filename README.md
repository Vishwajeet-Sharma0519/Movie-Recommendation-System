# 🎬 Recommendo - Premium Movie Recommendation System

Recommendo is a state-of-the-art, hybrid movie recommendation web application built on the popular **MovieLens 32M dataset** (containing 32 million ratings across 87,000+ movies). 

The project features a full machine learning pipeline in Jupyter Notebook and is deployed as an interactive web interface using **Streamlit**.

---

## 🚀 Key Features

* **IMDb Weighted Popularity (Demographic Filtering)**: Calculates globally high-rated movies using the IMDb weighted rating formula to account for vote count stability, solving the cold-start problem.
* **Content-Based Filtering**: Leverages TF-IDF vectorization and Cosine Similarity on combined movie genres and user-applied tags (21,000+ vocabulary words) to suggest highly similar movies.
* **Collaborative Filtering (SVD)**: Decomposes a centered, sparse user-movie ratings matrix using Singular Value Decomposition (SVD) to learn 20 latent preference factors.
* **Hybrid Personalized Recommendation Engine**: Combines content similarity and collaborative filtering. It finds movies similar to a target title and re-ranks them based on SVD rating predictions personalized for the selected user profile.
* **Premium Dark UI**: Built with custom Glassmorphism CSS styling, modern typography, and responsive cards.

---

## 📁 Repository Structure

```text
├── Rec_sys.ipynb         # Jupyter Notebook with the full ML pipeline
├── app.py                # Streamlit Web Application
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes massive raw datasets from git tracking
├── README.md             # Project documentation
├── svd_recommender.joblib # Saved SVD model factors
├── tfidf_vectorizer.joblib# Saved TF-IDF config
├── tfidf_matrix.joblib    # Saved TF-IDF vectors
└── movies_catalog.joblib # Processed movies catalog
```

---

## 🛠️ Local Installation & Setup

To run the web application locally:

### 1. Clone this repository
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
Make sure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
Start the Streamlit development server:
```bash
streamlit run app.py
```
This will automatically open your default browser to **`http://localhost:8501`**.

---

## 📈 Model Training & Dataset Info

If you wish to re-train the models using the Jupyter Notebook:
1. Download the `ml-32m` zip file from [GroupLens Research](https://grouplens.org/datasets/movielens/32m/).
2. Extract the CSV files (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) into a folder named `ml-32m` in the root of this project.
3. Open `Rec_sys.ipynb` in your Jupyter editor and run all cells. Running the final cells will automatically overwrite the `.joblib` model files with updated weights.

---

## 🛠️ Built With

* **Pandas & NumPy** - Data cleaning and matrix manipulation.
* **SciPy** - Sparse matrices (`csr_matrix`) and SVD calculations (`svds`).
* **Scikit-Learn** - TF-IDF vectorization and Cosine Similarity.
* **Streamlit** - High-performance web UI.
* **Joblib** - Serialization of trained models.
