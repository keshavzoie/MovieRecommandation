import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
from surprise import accuracy

# Load the MovieLens dataset (small version)
ratings = pd.read_csv("https://raw.githubusercontent.com/grouplens/datasets-100k/main/ml-latest-small/ratings.csv")
movies = pd.read_csv("https://raw.githubusercontent.com/grouplens/datasets-100k/main/ml-latest-small/movies.csv")

# Merge ratings with movie titles
df = ratings.merge(movies, on='movieId')

# Use Surprise's Reader class to define the rating scale
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split the data into train and test set
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use Singular Value Decomposition (SVD) for collaborative filtering
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate performance
rmse = accuracy.rmse(predictions)
print(f"Root Mean Squared Error: {rmse}")

# Function to recommend movies for a user
def recommend_movies(user_id, num_recommendations=5):
    movie_ids = df['movieId'].unique()
    user_movies = df[df['userId'] == user_id]['movieId'].tolist()
    
    # Predict ratings for all movies the user hasn't rated yet
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids if movie_id not in user_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top recommended movies
    top_movies = [pred.iid for pred in predictions[:num_recommendations]]
    
    return movies[movies['movieId'].isin(top_movies)]

# Get recommendations for a user
user_id = 1  # Example user ID
recommended_movies = recommend_movies(user_id)
print("Recommended Movies:")
print(recommended_movies[['movieId', 'title']])
