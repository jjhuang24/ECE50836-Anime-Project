import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn(query_point, k, features, anime_data):
    distances = []
    for i, point in enumerate(features):
        distance = euclidean_distance(query_point, point)
        distances.append((distance, i))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[1:k+1]  # Exclude the anime that was inputted (for obvious reasons)
    
    neighbor_indices = [idx for _, idx in neighbors]
    recommendations = anime_data.iloc[neighbor_indices][['name', 'genre', 'rating']]
    recommendations['distance'] = [dist for dist, _ in neighbors]
    
    return recommendations

def recommend_anime(anime_name, k=10):
    anime_index = anime_data[anime_data['name'] == anime_name].index
    if len(anime_index) == 0:
        return "Anime not found in the dataset."
    
    anime_index = anime_index[0]
    query_point = features_normalized[anime_index]
    
    recommendations = knn(query_point, k, features_normalized, anime_data)
    return recommendations

if __name__=="__main__":
    # Load the dataset using pandas
    anime_data = pd.read_csv('anime.csv')

    # Preprocess the data
    anime_data['genre'] = anime_data['genre'].fillna('').apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(anime_data['genre'])

    # Prepare features (genre and rating)
    features = np.hstack((genre_encoded, anime_data[['rating']].fillna(anime_data['rating'].mean()).values))

    features_normalized = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

    anime_name = input(f"Enter current anime: ")
    recommendations = recommend_anime(anime_name)

    print(f"Recommendations for {anime_name}:")
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {rec['name']} (Genre: {', '.join(rec['genre'])}, Rating: {rec['rating']:.2f})")