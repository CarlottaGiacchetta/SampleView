import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans



class Kmeans:


    def __init__(self, df, corpus):        
        self.df = df
        self.corpus = corpus

    
    #FUNZIONE CHE FA PCA -> SI PRENDONO LE COMPONENTI PRINCIPALI CHE RIPRODUCONO ALMENO IL 95% DI VARIANZA CUMULATA
    def pca(self, df):
        from sklearn.decomposition import PCA



        pca = PCA()
        pca.fit(df)
        variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_) 
        threshold = 0.95
        num_components = np.argmax(variance_ratio_cumulative >= threshold) + 1
        pca_selected = PCA(n_components=num_components) 
        data_transformed = pca_selected.fit_transform(df) 
        df_pca = pd.DataFrame(data=data_transformed , columns=['PC{}'.format(i) for i in range(1, num_components + 1)], index=df.index)


        return df_pca
    

    #FUNZIONE AUSILIARE CLUSTER_KMEANS -> PRENDE IN INPUT IL CORPUS E IL DATA SET DELLE COMONENTI PRINCIPALI.
    def clustering(self, df_pca, corpus):

        sse = []

        k_values = [2,3,4,5,6,7,8,9,10,12,13,14,15]
    

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init='auto')
            kmeans.fit_predict(df_pca)
            sse.append(kmeans.inertia_)
        kn = KneeLocator(k_values,sse, curve='convex', direction='decreasing')

        kmeans = KMeans(n_clusters=kn.knee, random_state=42,n_init='auto')
        kmeans.fit(df_pca) 
        cluster_labels = kmeans.predict(df_pca)
        
        corpus['cluster']=cluster_labels


        return corpus
    
    #FUNZIONE CHE GENERA I CLUSTER USANDO IL KMEANS. PRENDE IN INPUT LA METRICE DEGLI EMBEDDED DOCUMENTS E IL CORPUS.
    def cluster_KMEANS(self):
        df = self.df
        corpus = self.corpus
        df_pca = self.pca(df)
        corpus = self.clustering(df_pca,corpus)


        return corpus