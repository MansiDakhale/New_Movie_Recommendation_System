import pandas as pd

def load_data(filepath, nrows=100000):  # You can specify the number of rows to load
    ratings = pd.read_csv(r"C:\Users\OMEN\Desktop\DVC\New_MRS\datasets\ratings.csv", nrows=nrows)
    ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
    return ratings

def create_user_movie_matrix(ratings):
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')
    user_movie_matrix.fillna(0, inplace=True)
    return user_movie_matrix
