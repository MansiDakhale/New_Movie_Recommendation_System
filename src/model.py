from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

def compute_user_similarity(user_movie_matrix):
    similarity = cosine_similarity(user_movie_matrix)
    return pd.DataFrame(similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def compute_movie_similarity(user_movie_matrix):
    similarity = cosine_similarity(user_movie_matrix.T)
    return pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)
