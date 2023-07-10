import pandas as pd
from minisom import MiniSom
from typing import Tuple, Any, Dict
from sklearn.cluster import KMeans

from clustering.utils import get_best_n_clusters


def _kmeans_clustering(data: pd.DataFrame, n_clusters: int = None, seed: int = None) -> Tuple[Any, dict[str, int]]:

    if not n_clusters:
        n_clusters = get_best_n_clusters(data, seed)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    kmeans.fit(data)
    cluster_labels = kmeans.predict(data)
    return cluster_labels, {'algo': 'kmeans', 'n_clusters': n_clusters, 'seed': seed}


def _som_clustering(data: pd.DataFrame, map_size: Tuple[int, int] = (5, 5),
                    n_epochs: int = 100, seed: int = None) -> Tuple[list[Any], dict[str, int | tuple[int, int]]]:
    # Dimensioni della mappa SOM -> ne abbiamo scelta una di 25 celle in modo tale da avere 25 potenziali clusters.
    data = data.values
    # Si dice la dimensione della som, la grandezza dei vettori della som, la forma delle celle
    som = MiniSom(map_size[0], map_size[1], data.shape[1], topology='hexagonal', random_seed=seed)
    som.train(data, n_epochs)

    winners = {}
    for ind, input_vector in enumerate(data):
        winner = som.winner(input_vector)
        winners[ind] = winner

    # Ottieni i valori unici dal dizionario
    value_enum = {value: enum for enum, value in enumerate(set(winners.values()))}

    # Mappa i valori del dizionario originale all'enumerazione dei valori unici
    mapped_dict = {key: value_enum[value] for key, value in winners.items()}

    labels = [value for key, value in sorted(mapped_dict.items())]

    return labels, {'algo': 'SOM', 'n_clusters': len(set(labels)),
                    'map_size': map_size, 'n_epochs': n_epochs, 'seed': seed}


def perform_cluster(data: pd.DataFrame,
                    cluster_algo: str,
                    **kwargs) -> Tuple[list[int], Dict[str, int | tuple[int, int]]]:

    if cluster_algo == 'kmeans':
        clu_parameters = _kmeans_clustering.__code__.co_varnames[:_kmeans_clustering.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in clu_parameters}
        labels, clu_parameters = _kmeans_clustering(data, **kwargs)
    elif cluster_algo == 'SOM':
        clu_parameters = _som_clustering.__code__.co_varnames[:_som_clustering.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in clu_parameters}
        labels, clu_parameters = _som_clustering(data, **kwargs)
    else:
        raise ValueError("Invalid clustering algorithm. Try with kmeans o SOM")

    return labels, clu_parameters
