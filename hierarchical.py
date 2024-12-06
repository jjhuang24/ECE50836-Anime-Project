import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def agglomerative_clustering(features, n_clusters=10):
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = agglomerative.fit_predict(features)
    return clusters


def plot_dendrogram(features, method='ward'):
    linked = linkage(features, method=method)
    plt.figure(figsize=(12, 8))
    dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=False)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.savefig("Dendogram.png")
    plt.close()

anime_data = pd.read_csv('anime.csv')

anime_data['genre'] = anime_data['genre'].fillna('').apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(anime_data['genre'])

features = np.hstack((genre_encoded, anime_data[['rating']].fillna(anime_data['rating'].mean()).values))

scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

clusters = agglomerative_clustering(features_normalized, n_clusters=10)

anime_data['cluster'] = clusters

print(f"Cluster labels assigned:\n{anime_data['cluster'].value_counts()}")

plot_dendrogram(features_normalized, method='ward')

anime_data.to_csv('anime_clusters_hier.csv', index=False)