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


if __name__=="__main__":
    anime_data = pd.read_csv('anime.csv')

    anime_data['genre'] = anime_data['genre'].fillna('').apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(anime_data['genre'])

    features = np.hstack((genre_encoded, anime_data[['rating']].fillna(anime_data['rating'].mean()).values))

    features_normalized = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))

    clusters, centroids = kmeans(features_normalized, k = 10)
    
    anime_data['cluster'] = clusters
    
    print(f"Cluster labels assigned:\n{anime_data['cluster'].value_counts()}")
    
    anime_data.to_csv('anime_clusters_kmeans.csv', index=False)

