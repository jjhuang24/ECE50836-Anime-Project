import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

anime_df = pd.read_csv('anime.csv')

# Handle missing genres and transform genre column
anime_df['genre'] = anime_df['genre'].fillna('Unknown')
anime_df['genre'] = anime_df['genre'].str.split(',')

# Encode genres
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(anime_df['genre']),
                             columns=mlb.classes_,
                             index=anime_df.index)

# Concatenate features and drop NaNs
features = pd.concat([genre_encoded, anime_df['rating']], axis=1)
features = features.dropna()

# Reset the index of features to avoid index mismatch
features = features.reset_index(drop=True)
anime_clustered = anime_df.loc[features.index].reset_index(drop=True)

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply spectral clustering
n_clusters = 10
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
cluster_labels = spectral.fit_predict(features_scaled)

# Add cluster labels to the anime data
anime_clustered['Cluster'] = cluster_labels

# Recommendation function
def get_recommendations(anime_title, n=10):
    anime_info = anime_clustered[anime_clustered['name'] == anime_title]
    
    if anime_info.empty:
        return "Anime not found in the dataset."
    
    # Get the cluster and similar animes within the cluster
    cluster = anime_info['Cluster'].values[0]
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    
    # Calculate similarity within the cluster
    cluster_indices = cluster_animes.index
    cluster_features = features_scaled[cluster_indices]
    similarities = cosine_similarity(cluster_features)
    
    # Find similar animes
    anime_index = cluster_animes.index.get_loc(anime_info.index[0])
    similar_indices = similarities[anime_index].argsort()[::-1][1:n+1]
    recommendations = cluster_animes.iloc[similar_indices]
    
    return recommendations[['name', 'genre', 'rating']]

# Input for recommendations
input_anime = input(f"Enter current anime: ")
recommendations = get_recommendations(input_anime)

# Output recommendations
print(f"Top 10 recommendations based on {input_anime} using Spectral Clustering:")
for i, (_, row) in enumerate(recommendations.iterrows(), 1):
    print(f"{i}. {row['name']}")
    print(f"   Genres: {', '.join(row['genre'])}")
    print(f"   Rating: {row['rating']}")
    print()

# Cluster summaries
for cluster in range(n_clusters):
    cluster_animes = anime_clustered[anime_clustered['Cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Number of animes: {len(cluster_animes)}")
    print(f"Average rating: {cluster_animes['rating'].mean():.2f}")
    print("Top 5 genres:")
    top_genres = cluster_animes['genre'].explode().value_counts().head()
    for genre, count in top_genres.items():
        print(f"  - {genre}: {count}")

# Plot cluster sizes
cluster_sizes = anime_clustered['Cluster'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(cluster_sizes.index, cluster_sizes.values)
plt.title('Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Number of Anime')
plt.show()
