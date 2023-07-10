from clustering.algorithms import perform_cluster
from clustering.evaluation import evaluate_clusters
from clustering.utils import plot_k_means_clusters, plot_som_clusters, pca

from Embedding import perform_embedding

from typing import List, Dict
import pandas as pd
import numpy as np


class SampleView:
    """
    questa è la funzione principale che permette di i parametri di quelle successive.
    questa funzione prende in input:

    - corpus -> un data frame pandas
    - sample -> la grandezza del campione. Di default 1000
    - text-> una variabile di tipo 'object'. Di default None
    - topic -> una variabile di tipo 'object'. Di default None
    - Language -> una variabile di tipo 'object'. Di default None
    - time -> una variabile di tipo 'datetime64[ns]'. Di default None
    - vector_size -> la dimensione delle rappresentazioni vettoriali dei documenti -> di default 100
    - window -> la grandezza del contesto dell'algoritmo DOC2VEC. Di default 5
    - min_count -> il numero minimo di occorrenze di una parola. Di default 1
    - workers -> il numero di elementi di parallelizzazione dell'algoritmo.indica il numero di thread da utilizzare
        durante l'addestramento del modello.
      Questo parametro specifica il numero di thread paralleli che verranno utilizzati per accelerare il processo
      di addestramento e l'elaborazione dei documenti.
      Di default 1
    - emb_epochs -> numero di epoche di allenamento dell'algoritmo -> di default 10
    """

    def __init__(self, data):
        self.data = data

        self.seed = None
        self.frac = None
        self.sampling_var = None
        self.text_var = None
        self.emb_model_params = None
        self.vec = None
        self.variables = None
        self.clu_parameters = None
        self.data_sample = None

    def sample(self, frac: float, sampling_var: List[str], text_var: str = None,
               use_pca: bool = True, cluster_algo: str = 'kmeans', n_clusters: int = None,
               emb_algo: str = 'Doc2Vec', vector_size: int = 100, window: int = 5,
               min_count: int = 1, workers: int = 3, emb_epochs: int = 10,
               seed: int = 42, **kwargs):

        self.frac = frac
        self.seed = seed
        self.sampling_var = set(sampling_var)
        self.text_var = text_var

        if text_var:
            self.vec, self.emb_model_params = perform_embedding(self.data[text_var], emb_algo=emb_algo,
                                                                vector_size=vector_size, window=window,
                                                                min_count=min_count, workers=workers,
                                                                emb_epochs=emb_epochs, **kwargs)

            if use_pca:
                self.variables = pca(self.vec, self.seed)
            else:
                self.variables = self.vec


            self.data['cluster_labels'], self.clu_parameters = perform_cluster(data=self.variables,
                                                                               cluster_algo=cluster_algo,
                                                                               n_clusters=n_clusters,
                                                                               seed=seed)
            self.sampling_var.add('cluster_labels')
            self.sampling_var = list(self.sampling_var)

        self.data_sample = self.data.groupby(self.sampling_var).apply(
            lambda x: x.sample(int(round(x.shape[0] * frac, 0))))

        # self.sample, _ = train_test_split(self.data, train_size=frac, stratify=self.data[sampling_var],
        # random_state=seed)
        self.data_sample.drop('cluster_labels', axis=1, inplace=True)
        return self.data_sample

    def cluster_evaluation(self, metric: str = 'silhouette_score') -> float:
        return evaluate_clusters(self.vec.values, self.data['cluster_labels'], metric)

    def plot_clusters(self):
        if self.clu_parameters['algo'] == 'k-means':
            plot_k_means_clusters(self.variables, self.data['cluster_labels'])
        elif self.clu_parameters['algo'] == 'SOM':
            plot_som_clusters(self.data['cluster_labels'], self.clu_parameters['map_size'])


    def get_embeddings_params(self) -> Dict:
        return self.emb_model_params

    def get_embeddings_vec(self) -> np.ndarray:
        return self.vec.values

    def get_pca_variables(self) -> pd.DataFrame:
        return self.variables

    def get_clustering_params(self) -> Dict:
        return self.clu_parameters

    def get_clustering_labels(self) -> List[int]:
        return self.data['cluster_labels'].to_list()

    def __str__(self) -> str:
        s = f"+{'-' * 30}+\n" \
            f"|{'Sample summary':^30}|\n" \
            f"+{'-' * 30}+\n" \
            f"|{'Initial rows:':<14}{self.data.shape[0]:>12} row|\n"
        if self.sampling_var:
            s += f"|{'Sampling vars:':<25}{', '.join([x for x in self.sampling_var if x != 'cluster_labels']):>5}|\n"
        if self.text_var:
            s += f"|{'Sampling text var:':<19}{self.text_var:>11}|\n"
        if self.emb_model_params:
            s += f"+{'-' * 30}+\n" \
                 f"|{'Embeddings parameters':^30}|\n" \
                 f"+{'-' * 30}+\n"
            for p in self.emb_model_params:
                s += f"|{''.join([p, ':']):<20}{self.emb_model_params[p]:>10}|\n"
        if self.clu_parameters:
            s += f"+{'-' * 30}+\n" \
                 f"|{'Clustering parameters':^30}|\n" \
                 f"+{'-' * 30}+\n"
            for p in self.clu_parameters:
                s += f"|{''.join([p, ':']):<20}{self.clu_parameters[p]:>10}|\n"
        if isinstance(self.data_sample, pd.DataFrame):
            s += f"+{'-' * 30}+\n" \
                 f"|{'Sample Info':^30}|\n" \
                 f"+{'-' * 30}+\n" \
                 f"|{'Sample frac':<20}{self.frac:>10}|\n" \
                 f"|{'Sample rows number':<20}{self.data_sample.shape[0]:>10}|\n"
        s += f"+{'-' * 30}+\n"
        return s
