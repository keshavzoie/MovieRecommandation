
# 🎬 Movie Recommendation System using Collaborative Filtering

This project is a movie recommendation system built using **collaborative filtering** with **Singular Value Decomposition (SVD)**. It uses the MovieLens dataset and the `Surprise` library to predict and recommend movies based on user preferences.

## 📌 Features

- Collaborative Filtering (SVD)
- Personalized movie recommendations
- Evaluation using RMSE
- Uses MovieLens `ml-latest-small` dataset

## 🧠 Algorithm

We use **SVD (Singular Value Decomposition)** to learn latent factors of users and movies. This model is effective in capturing user preferences and providing high-quality recommendations.

## 🛠️ Requirements

Install the required libraries:

```bash
pip install pandas numpy scikit-learn scikit-surprise
```

## 📁 Dataset

We use the **MovieLens Latest Small Dataset**, which includes:

- `ratings.csv`: User ratings for movies
- `movies.csv`: Movie titles and genres

These are loaded directly from GitHub via raw links.

## 🚀 How It Works

1. **Data Loading & Merging**
   - Ratings and movie metadata are merged for analysis.

2. **Training the Model**
   - The dataset is converted for `Surprise` using `Reader`.
   - Model trained using `SVD`.

3. **Evaluation**
   - The model's predictions are tested on a split test set.
   - RMSE (Root Mean Squared Error) is used to evaluate accuracy.

4. **Recommendation Function**
   - Recommend top N movies for a given user, excluding already-rated ones.

## ✅ Output

Example console output:

```
Root Mean Squared Error: 0.8721
Recommended Movies:
     movieId                     title
123      5971   Like Stars on Earth (2007)
435      7361           Harry Potter (2001)
...
```

## 📦 File Structure

```
movie_recommender.py    # Main project code
README.md               # Project documentation
```

## 🧑‍💻 Example Usage

Run the script and it will:

- Train the model
- Evaluate with RMSE
- Recommend top 5 movies for user ID `1`

## 🔍 Future Improvements

- Add a web interface using Flask or Streamlit
- Use item-based collaborative filtering for comparison
- Include genres and metadata for hybrid recommendations

## 📚 References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Surprise Library](http://surpriselib.com/)
