import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def initialize_centroids(features, k):
    np.random.seed(42)
    indices = np.random.choice(features.shape[0], k, replace=False)
    return features[indices]

def assign_clusters(features, centroids):
    distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def compute_centroids(features, clusters, k):
    centroids = np.array([features[clusters == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(features, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(features, k)
    for _ in range(max_iters):
        clusters = assign_clusters(features, centroids)
        new_centroids = compute_centroids(features, clusters, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return clusters, centroids

def recommend_anime(anime_name, n_recommendations=10):
    anime_index = anime_data[anime_data['name'] == anime_name].index
    if len(anime_index) == 0:
        return "Anime not found in the dataset."
    
    anime_index = anime_index[0]
    anime_cluster = clusters[anime_index]
    
    cluster_anime = anime_data[clusters == anime_cluster]
    distances = np.linalg.norm(features_normalized[clusters == anime_cluster] - features_normalized[anime_index], axis=1)
    
    recommendations = cluster_anime.iloc[np.argsort(distances)[1:n_recommendations+1]]
    return recommendations[['name', 'genre', 'rating']]

if __name__=="__main__":
    anime_data = pd.read_csv('anime.csv')

    anime_data['genre'] = anime_data['genre'].fillna('').apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(anime_data['genre'])

    # Prepare features (genre and rating)
    features = np.hstack((genre_encoded, anime_data[['rating']].fillna(anime_data['rating'].mean()).values))

    features_normalized = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

    clusters, centroids = kmeans(features_normalized, 10)

    anime_name = input(f"Enter current anime: ")
    recommendations = recommend_anime(anime_name)

    print(f"Recommendations for {anime_name}:")
    for i, (_, rec) in enumerate(recommendations.iterrows(), 1):
        print(f"{i}. {rec['name']} (Genre: {', '.join(rec['genre'])}, Rating: {rec['rating']:.2f})")