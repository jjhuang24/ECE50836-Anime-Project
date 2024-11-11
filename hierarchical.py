import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        distances = self._calculate_distances(X)
        
        for _ in range(n_samples - self.n_clusters):
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            
            self.labels_[self.labels_ == self.labels_[j]] = self.labels_[i]
            
            distances[i] = np.minimum(distances[i], distances[j])
            distances[:, i] = distances[i]
            distances[j, :] = np.inf
            distances[:, j] = np.inf
            distances[i, i] = np.inf
        
        unique_labels = np.unique(self.labels_)
        for i, label in enumerate(unique_labels):
            self.labels_[self.labels_ == label] = i

    def _calculate_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[i, j] = distances[j, i] = np.linalg.norm(X[i] - X[j])
        return distances

anime = pd.read_csv('anime.csv')
anime['genre'] = anime['genre'].fillna('Unknown').str.split(',')

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(anime['genre']),
                             columns=mlb.classes_,
                             index=anime.index)

features = pd.concat([genre_encoded, anime['rating']], axis=1)
features = features.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

n_clusters = 10
clustering = HierarchicalClustering(n_clusters=n_clusters)
clustering.fit(features_scaled)

anime_clustered = anime.loc[features.index].copy()
anime_clustered['Cluster'] = clustering.labels_

def get_recommendations(anime_title, n=10):
    anime_info = anime_clustered[anime_clustered['name'] == anime_title]
    
    if anime_info.empty:
        return "Anime not found"
    
    cluster = anime_info['Cluster'].values[0]
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    
    recommendations = cluster_animes[cluster_animes['name'] != anime_title].sort_values('rating', ascending=False)
    return recommendations.head(n)[['name', 'genre', 'rating']]

input_anime = input("Enter current anime: ")
recommendations = get_recommendations(input_anime)

print(f"\nTop 10 recommendations based on {input_anime} using custom hierarchical clustering:")
for i, (_, row) in enumerate(recommendations.iterrows(), 1):
    print(f"{i}. {row['name']} (Genre: {', '.join(row['genre'])}, Rating: {row['rating']:.2f})")

for cluster in range(n_clusters):
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    print(f"\nCluster {cluster+1}:")
    print(f"Number of animes: {len(cluster_animes)}")
    print(f"Average rating: {cluster_animes['rating'].mean():.2f}")
    print("Top 5 genres:")
    top_genres = cluster_animes['genre'].explode().value_counts().head()
    for genre, count in top_genres.items():
        print(f"  - {genre}: {count}")

plt.figure(figsize=(10, 7))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=clustering.labels_, cmap='viridis')
plt.title('Hierarchical Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.savefig("Clustering_Results.png")
plt.close()