import pandas as pd
from minisom import MiniSom
import numpy as np


class SOM:


    def __init__(self, df, corpus):
        self.df = df
        self.corpus = corpus


    def cluster_som(self):


        df = self.df

        corpus = self.corpus
        data = df.values

        map_size = (5,5)  # Dimensioni della mappa SOM -> ne abbiamo scelta una di 25 celle in modo tale da avere 25 potenziali clusters.
        n_epochs = 100  
        som = MiniSom(map_size[0], map_size[1], data.shape[1], topology='hexagonal') #input funzione che inizializza la som. Si dice la dimensione della som, la grandezza dei vettori della som, la forma delle celle (in questo caso esagonali)
        som.train(data, n_epochs) 
        map_grid = som.get_weights()  

        winners={}
        
        for input_vector in data:
            winner = som.winner(input_vector)
            winners[winner]=map_grid[winner]

        df_w = pd.DataFrame.from_dict(winners, orient='index') #un data frame che contenga le informazioni sui codebook che contengono almeno un'unità statistica -> cella - vettore
        df_w=self.clustering_som(df_w) #si associa un etichetta di un cluster a queste celle
        corpus['cluster'] = np.apply_along_axis(self.associate_cluster, axis=1, arr=data, df=df_w, som =som) #si associa ai documenti il rispettivo cluster
    
        return corpus
    
    
    #FUNZIONE AUSILIARE PER CLUSTER_SOM -> ASSOCIA OGNI CELLA A UN CLUSTER
    def clustering_som(self, data):
        data['cluster'] = range(len(data))
        return data
    


    #FUNZIONE AUSILIARE PER CLUSTER_SOM -> ASSOCIA OGNI DOCUMENTO AL CLUSTER CORRISPONDENTE

    
    def associate_cluster(self, input_vector, df, som):

        winner = som.winner(input_vector)
        ser = pd.Series(df.loc[[winner]]['cluster'])
        valore = int(ser.iloc[0])
        
        return valore

