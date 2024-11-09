import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def compute_similarity_matrix(X):
    return cosine_similarity(X)

def compute_laplacian_matrix(S):
    D = np.diag(S.sum(axis=1))
    L = D - S
    
    return L

def compute_eigenvectors(L, n_clusters):
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    idx = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvectors[:, :n_clusters]

def spectral_clustering(X, n_clusters):
    S = compute_similarity_matrix(X)
    L = compute_laplacian_matrix(S)
    
    eigenvectors = compute_eigenvectors(L, n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(eigenvectors)
    
    return cluster_labels

anime_df = pd.read_csv('anime.csv')

anime_df['genre'] = anime_df['genre'].fillna('Unknown')
anime_df['genre'] = anime_df['genre'].str.split(',')

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

for cluster in range(n_clusters):
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Number of animes: {len(cluster_animes)}")
    print(f"Average rating: {cluster_animes['rating'].mean():.2f}")
    print("Top 5 genres:")
    top_genres = cluster_animes['genre'].explode().value_counts().head()
    for genre, count in top_genres.items():
        print(f"  - {genre}: {count}")
