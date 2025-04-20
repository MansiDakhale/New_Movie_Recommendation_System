import numpy as np

def predict_rating(user_id, movie_id, user_movie_matrix, user_similarity, k=5):
    if movie_id not in user_movie_matrix.columns or user_id not in user_movie_matrix.index:
        return np.nan

    sim_scores = user_similarity[user_id]
    movie_ratings = user_movie_matrix[movie_id]
    
    valid_users = movie_ratings[movie_ratings > 0].index
    sim_scores = sim_scores.loc[valid_users]
    movie_ratings = movie_ratings.loc[valid_users]

    top_k_users = sim_scores.sort_values(ascending=False).head(k)
    top_k_ratings = movie_ratings.loc[top_k_users.index]

    if top_k_users.sum() == 0:
        return 0

    return np.dot(top_k_users.values, top_k_ratings.values) / top_k_users.sum()

def recommend_movies(user_id, user_movie_matrix, user_similarity, n=5):
    rated_movies = user_movie_matrix.loc[user_id]
    unrated_movies = rated_movies[rated_movies == 0].index

    predicted_ratings = {}
    for movie_id in unrated_movies:
        predicted = predict_rating(user_id, movie_id, user_movie_matrix, user_similarity)
        predicted_ratings[movie_id] = predicted

    top_n = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:n]
    return top_n
