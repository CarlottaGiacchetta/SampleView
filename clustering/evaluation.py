
import numpy as np
from typing import List
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score


def _dunn_index(data: np.ndarray, labels: List[int]) -> float:
    unique_labels = np.unique(labels)

    # Calcola la distanza massima tra i punti all'interno di ogni cluster
    intra_cluster_distances = np.array(
        [np.max(cdist(data[labels == label], data[labels == label])) for label in unique_labels])

    # Calcola la distanza minima tra i centroidi dei cluster
    centroid_distances = cdist(
        np.array([np.mean(data[labels == label], axis=0) for label in unique_labels]),
        np.array([np.mean(data[labels == other_label], axis=0) for other_label in unique_labels]))

    # Calcola l'Indice di Dunn
    min_inter_cluster_distance = np.min(centroid_distances[np.nonzero(centroid_distances)])
    max_intra_cluster_distance = np.max(intra_cluster_distances)
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    return dunn_index


def evaluate_clusters(data: np.ndarray, labels: List[int], metric: str) -> float:
    if metric == 'silhouette_score':
        score = silhouette_score(data, labels)
    elif metric == 'dunn':
        score = _dunn_index(data, labels)
    else:
        raise ValueError("Invalid evaluation metrics. Try with silhouette_score")
    return score
