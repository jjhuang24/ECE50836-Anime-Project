import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import DBSCAN

def dbscan_clustering(features, eps=0.5, min_samples=50):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    clusters = dbscan.fit_predict(features)
    return clusters

if __name__ == "__main__":
    anime_data = pd.read_csv('anime.csv')

    anime_data['genre'] = anime_data['genre'].fillna('').apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(anime_data['genre'])

    features = np.hstack((genre_encoded, anime_data[['rating']].fillna(anime_data['rating'].mean()).values))

    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    clusters = dbscan_clustering(features_normalized, eps=10, min_samples=5)

    anime_data['cluster'] = clusters

    print(f"Cluster labels assigned:\n{anime_data['cluster'].value_counts()}")

    anime_data.to_csv('anime_clusters_dbscan.csv', index=False)
