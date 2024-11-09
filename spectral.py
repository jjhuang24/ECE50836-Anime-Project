import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def compute_similarity_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return np.dot(X, X.T) / (norm * norm.T)

def compute_laplacian_matrix(S):
    D = np.diag(np.sum(S, axis=1))
    return D - S

def compute_eigenvectors(L, n_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)[:n_clusters]
    return eigenvectors[:, idx]

def initialize_centroids(features, k):
    return features[np.random.choice(features.shape[0], k, replace=False)]

def assign_clusters(features, centroids):
    distances = np.sqrt(((features[:, np.newaxis, :] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)

def compute_centroids(features, clusters, k):
    return np.array([features[clusters == i].mean(axis=0) for i in range(k)])

def kmeans(features, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(features, k)
    for _ in range(max_iters):
        clusters = assign_clusters(features, centroids)
        new_centroids = compute_centroids(features, clusters, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return clusters, centroids

def spectral_clustering(X, n_clusters):
    S = compute_similarity_matrix(X)
    L = compute_laplacian_matrix(S)
    eigenvectors = compute_eigenvectors(L, n_clusters)
    clusters, _ = kmeans(eigenvectors, n_clusters)
    return clusters

anime_df = pd.read_csv('anime.csv')
anime_df['genre'] = anime_df['genre'].fillna('Unknown').str.split(',')

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(anime_df['genre']),
                             columns=mlb.classes_,
                             index=anime_df.index)

features = pd.concat([genre_encoded, anime_df['rating']], axis=1)
features = features.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

n_clusters = 10
cluster_labels = spectral_clustering(features_scaled, n_clusters)

anime_clustered = anime_df.loc[features.index].copy()
anime_clustered['Cluster'] = cluster_labels

def get_recommendations(anime_name, anime_data, n_recommendations=10):
    if anime_name not in anime_data['name'].values:
        return f'Anime {anime_name} not found in the dataset.'

    anime_row = anime_data[anime_data['name'] == anime_name]
    cluster = anime_row['Cluster'].values[0]

    recommendations = anime_data[anime_data['Cluster'] == cluster]
    recommendations = recommendations[recommendations['name'] != anime_name]

    return recommendations[['name', 'rating']].sort_values('rating', ascending=False).head(n_recommendations).to_dict('records')

input_anime = input(f"Enter current anime: ")
recommendations = get_recommendations(input_anime, anime_clustered)

print(f"\nTop 10 recommendations for {input_anime}:")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec['name']} (Rating: {rec['rating']:.2f})")

for cluster in range(n_clusters):
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Number of animes: {len(cluster_animes)}")
    print(f"Average rating: {cluster_animes['rating'].mean():.2f}")
    print("Top 5 genres:")
    top_genres = cluster_animes['genre'].explode().value_counts().head()
    for genre, count in top_genres.items():
        print(f"  - {genre}: {count}")