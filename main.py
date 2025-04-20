from src.data_loader import load_data, create_user_movie_matrix
from src.model import compute_user_similarity
from src.recommender import recommend_movies

def main():
    ratings = load_data(r"C:\Users\OMEN\Desktop\DVC\New_MRS\datasets\ratings.csv")
    matrix = create_user_movie_matrix(ratings)
    similarity = compute_user_similarity(matrix)

    user_id = 1
    recommendations = recommend_movies(user_id, matrix, similarity, n=5)

    print(f"Top 5 movie recommendations for user {user_id}:")
    for movie_id, score in recommendations:
        print(f"Movie ID: {movie_id}, Predicted Rating: {round(score, 2)}")

if __name__ == "__main__":
    main()
