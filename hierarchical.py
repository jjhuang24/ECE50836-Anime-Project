import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

anime = pd.read_csv('anime.csv')

anime['genre'] = anime['genre'].fillna('Unknown')
anime['genre'] = anime['genre'].str.split(',')

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(anime['genre']),
                             columns=mlb.classes_,
                             index=anime.index)

features = pd.concat([genre_encoded, anime['rating']], axis=1)

features = features.dropna()

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

n_clusters = 10
clustering = AgglomerativeClustering(n_clusters=n_clusters)
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

input_anime =  input(f"Enter current anime: ")
recommendations = get_recommendations(input_anime)

print(f"Top 10 recommendations based on {input_anime} using hierarchical clustering:")
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
dendrogram(linkage(features_scaled[:100], method='ward')) 
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
plt.savefig("Dendogram")