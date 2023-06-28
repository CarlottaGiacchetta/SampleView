import pandas as pd
from minisom import MiniSom
from typing import Tuple, List
from numpy import cumsum, argmax
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def _pca(df: pd.DataFrame, var_threshold: float = 0.95) -> pd.DataFrame:
    # FUNZIONE CHE FA PCA -> SI PRENDONO LE COMPONENTI PRINCIPALI CHE RIPRODUCONO ALMENO IL 95% DI VARIANZA CUMULATA
    pca_model = PCA()
    pca_model.fit(df)
    variance_ratio_cumulative = cumsum(pca_model.explained_variance_ratio_)
    num_components = argmax(variance_ratio_cumulative >= var_threshold) + 1
    pca_selected = PCA(n_components=num_components)
    data_transformed = pca_selected.fit_transform(df)
    df_pca = pd.DataFrame(data=data_transformed,
                          columns=[f'PC_{i}' for i in range(1, num_components + 1)],
                          index=df.index)
    return df_pca


def _get_best_n_cluster(data: pd.DataFrame, seed: int = 42) -> int:
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


def _kmeans_clustering(data: pd.DataFrame, use_pca: bool = True, n_cluster: int = None, seed: int = 42) -> List[int]:
    if use_pca:
        data = _pca(data)
        print(data)
    if not n_cluster:
        n_cluster = _get_best_n_cluster(data, seed)

    kmeans = KMeans(n_clusters=n_cluster, random_state=seed, n_init='auto')
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)
    return cluster_labels


def _som_clustering(data: pd.DataFrame, map_size: Tuple[int, int] = (5, 5), n_epochs: int = 100) -> List[int]:
    # Dimensioni della mappa SOM -> ne abbiamo scelta una di 25 celle in modo tale da avere 25 potenziali clusters.
    data = data.values
    # Si dice la dimensione della som, la grandezza dei vettori della som, la forma delle celle
    som = MiniSom(map_size[0], map_size[1], data.shape[1], topology='hexagonal', random_seed=42)
    som.train(data, n_epochs)

    winners = {}
    for ind, input_vector in enumerate(data):
        winner = som.winner(input_vector)
        winners[ind] = winner

    # Ottieni i valori unici dal dizionario
    value_enum = {value: enum for enum, value in enumerate(set(winners.values()))}

    # Mappa i valori del dizionario originale all'enumerazione dei valori unici
    mapped_dict = {key: value_enum[value] for key, value in winners.items()}

    return [value for key, value in sorted(mapped_dict.items())]


def perform_cluster(data: pd.DataFrame, cluster_algo: str, **kwargs) -> pd.DataFrame:
    if cluster_algo == 'kmeans':
        kmeans_parameters = _kmeans_clustering.__code__.co_varnames[:_kmeans_clustering.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in kmeans_parameters}
        label = _kmeans_clustering(data, **kwargs)
    elif cluster_algo == 'SOM':
        som_parameters = _som_clustering.__code__.co_varnames[:_som_clustering.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in som_parameters}
        label = _som_clustering(data, **kwargs)
    else:
        raise ValueError("Invalid clustering algorithm")

    data['cluster'] = label
    return data
