import pandas as pd
import numpy as np
from numpy import cumsum, argmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt


def pca(df: pd.DataFrame, seed, var_threshold: float = 0.95) -> pd.DataFrame:
    # FUNZIONE CHE FA PCA -> SI PRENDONO LE COMPONENTI PRINCIPALI CHE RIPRODUCONO ALMENO IL 95% DI VARIANZA CUMULATA
    pca_model = PCA(random_state=seed)
    pca_model.fit(df)
    variance_ratio_cumulative = cumsum(pca_model.explained_variance_ratio_)
    num_components = argmax(variance_ratio_cumulative >= var_threshold) + 1
    pca_selected = PCA(n_components=num_components, random_state=seed)
    data_transformed = pca_selected.fit_transform(df)
    df_pca = pd.DataFrame(data=data_transformed,
                          columns=[f'PC_{i}' for i in range(1, num_components + 1)],
                          index=df.index)
    return df_pca


def plot_clusters(df: pd.DataFrame, labels: list) -> None:
    X = df.values

    # Genera un insieme di colori unici per i cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.get_cmap('tab10', n_clusters)

    # Crea un grafico per ogni cluster
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(i), label=f'Cluster {label}')

    # Personalizza il grafico
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Clustering Plot')
    plt.legend()

    # Mostra il grafico
    plt.show()


def get_best_n_clusters(data: pd.DataFrame, seed: int = 42) -> int:
    max_n_clusters = 15
    max_score = -1  # Variabile per tenere traccia del punteggio massimo di silhouette
    best_n_clusters = 0  # Variabile per tenere traccia del numero di cluster ottimale

    for n_clusters in range(2, max_n_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        cluster_labels = kmeans.fit_predict(data)

        # Calcolo del coefficiente di silhouette score
        silhouette = silhouette_score(data, cluster_labels)

        # Verifica se il punteggio silhouette score è il massimo finora
        if silhouette > max_score:
            max_score = silhouette
            best_n_clusters = n_clusters

    return best_n_clusters
