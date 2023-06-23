
import pandas as pd
from minisom import MiniSom
import numpy as np
from kneed import KneeLocator
import re
from sklearn.cluster import KMeans
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases
from tqdm import tqdm
import time
tqdm.pandas()

class TextSampler:
    ''' 
    questa è la funzione principale che permette di settare i parametri di quelle successive. 
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
    - workers -> il numero di elementi di parallelizzazione dell'algoritmo.indica il numero di thread da utilizzare durante l'addestramento del modello.
      Questo parametro specifica il numero di thread paralleli che verranno utilizzati per accelerare il processo di addestramento e l'elaborazione dei documenti.
      Di default 1
    - emb_epochs -> numero di epoche di allenamento dell'algoritmo -> di default 10
    '''
    def __init__(self, corpus, sample=1000, text=None, topic=None, Language=None, time=None,
                 vector_size=100, window=5, min_count=1, workers = 1, emb_epochs=10, cluster = 'Kmeans'):
        self.corpus = corpus
        self.text = text
        self.cluster=cluster
        self.topic=topic
        self.Language = Language
        self.time=time
        self.vector_size=vector_size
        self.window = window
        self.min_count =min_count
        self.epochs=emb_epochs
        self.workers=workers
        self.sample = sample



    #FUNZIONE AUSILIARE DI PREPROCESSING
    def clear_text(self, t:str) -> list:



        if pd.isnull(t):

            t=''
        
        str(t)
        t = re.sub(r'[^a-zA-Z]+', ' ', t)
        t = t.lower()
        t = [w for w in t.split() if len(w) >1]
        if len(t)>0:
            return t
        else:
            return ['']
        
    #FUNZIONE CHE FA DOC2VEC
        
    def doc2Vec(self, data, text):

    
        # Pulizia dei testi
        


        lemmatizer = WordNetLemmatizer()
        data['clear_text'] = data[text].apply(lambda x: self.clear_text(x))
        data['clear_text'] = data['clear_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        bigram_model = Phrases(data['clear_text'],  min_count=5, threshold=0.2)
        data['clear_text'] = data['clear_text'].apply(lambda x: bigram_model[bigram_model[x]])

        
        #trasformo i miei documenti in tagged_docs
        tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(data['clear_text'])]

        # Creazione del modello Doc2vec
        model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        

        document_vectors = {} 

        for i in range(len(data)):
            document_vectors[i] = model.dv[i]
        df = pd.DataFrame.from_dict(document_vectors, orient='index')
        df = df.apply(lambda x: pd.Series(x), axis=1)
        return df
    


    #FUNZIONE AUSILIARE PER VARIABILE TIME
    def assegna_intervallo(self, valore, intervalli):
        for i, intervallo in enumerate(intervalli):
            if valore >= intervallo[0] and valore <= intervallo[1]:
                return intervallo
        return None


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


    #FUNZIONE CLUSTER_SOM CHE FA LA CLUSTER ANALYSIS USANDO LA SOM
    def cluster_SOM(self, df, corpus):
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
    def clustering(self, corpus, df_pca):
        print(corpus, df_pca)
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
    def cluster_KMEANS(self, corpus, df):

        df_pca = self.pca(df)
        corpus = self.clustering(corpus, df_pca)


        return corpus
    
    #FUNZIONE PRINCIPALE DI TEXT SAMPLE, CHE GENERA IL CAMPIONE. NON PRENDE IN INPUT NIENTE.
        
    def Sampler(self):
        corpus = self.corpus
        text = self.text
        cluster = self.cluster
        topic = self.topic
        Language = self.Language 
        time = self.time
        sample = self.sample

        if text!=None:
 

            if time != None: 

                min = corpus[time].min()
                max = corpus[time].max()
                ampiezza_intervallo = (max - min) / 5

                intervalli = []
                valore_inizio = min
                valore_fine = valore_inizio + ampiezza_intervallo

                for i in range(5):
                    intervalli.append((valore_inizio, valore_fine))
                    valore_inizio = valore_fine
                    valore_fine = valore_inizio + ampiezza_intervallo


                corpus['tquantile']=corpus[time].apply(lambda x: self.assegna_intervallo(x, intervalli))



            
            df = self.doc2Vec(corpus, text)
            if cluster=='SOM':
                corpus=self.cluster_SOM(df, corpus)

            elif cluster=='Kmeans':
                corpus=self.cluster_KMEANS(corpus, df)


        
        else:

            

            if time!=None:
                min = corpus[time].min()
                max = corpus[time].max()
                ampiezza_intervallo = (max - min) / 5
                intervalli = []
                valore_inizio = min
                valore_fine = valore_inizio + ampiezza_intervallo

                for i in range(5):
                    intervalli.append((valore_inizio, valore_fine))
                    valore_inizio = valore_fine
                    valore_fine = valore_inizio + ampiezza_intervallo


                corpus['tquantile']=corpus[time].apply(lambda x: self.assegna_intervallo(x, intervalli))


                        


        if topic!=None and text!=None and Language == None and time== None:
            dfNew=pd.DataFrame() 



            dfNew = corpus.groupby(['cluster', topic ]).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                cluster = row['cluster']
                top= row[topic]
                num_samples = int(row['category_sample'])
                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[topic]==top)].drop(['clear_text','cluster'], axis=1)

                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])
           

            return campione
        
        elif Language!=None and text!=None  and topic == None and time== None:
            dfNew=pd.DataFrame() 



            dfNew = corpus.groupby(['cluster', Language ]).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                cluster = row['cluster']
                lang= row[Language]
                num_samples = row['category_sample']
                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[Language]==lang)].drop(['clear_text', 'cluster'], axis=1)
                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])
     

            return campione
        elif topic!=None and text!=None and Language!=None and time== None:
            dfNew=pd.DataFrame() 



            dfNew = corpus.groupby(['cluster', topic, Language]).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                cluster = row['cluster']
                top= row[topic]
                lang = row[Language]
                num_samples = int(row['category_sample'])

                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[topic]==top) & (corpus[Language]==lang)].drop(['clear_text', 'cluster'], axis=1)

                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])
   

            return campione
        
        elif topic!=None and text!=None and Language!=None and time != None:
            dfNew=pd.DataFrame() 
            dfNew = corpus.groupby(['cluster', topic, Language, 'tquantile']).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                tquant = row['tquantile']
                cluster = row['cluster']
                top= row[topic]
                lang = row[Language]
                num_samples = int(row['category_sample'])

                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[topic]==top) & (corpus[Language]==lang)  & (corpus['tquantile']==tquant)].drop(['clear_text', 'tquantile', 'cluster'], axis=1)

                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])

            return campione
        elif topic!=None and text!=None and Language==None and time != None:
            dfNew=pd.DataFrame() 
            dfNew = corpus.groupby(['cluster', topic, 'tquantile']).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                tquant = row['tquantile']
                cluster = row['cluster']
                top= row[topic]
                num_samples = int(row['category_sample'])

                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[topic]==top) & (corpus['tquantile']==tquant)].drop(['clear_text', 'tquantile', 'cluster'], axis=1)

                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])
  

            return campione
        elif topic==None and text!=None and Language!=None and time != None:
            dfNew=pd.DataFrame() 
            dfNew = corpus.groupby(['cluster', Language, 'tquantile']).size().reset_index(name='Frequencies')
    
            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                tquant = row['tquantile']
                cluster = row['cluster']
                lang = row[Language]
                num_samples = int(row['category_sample'])

                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus[Language]==lang)  & (corpus['tquantile']==tquant)].drop(['clear_text', 'tquantile', 'cluster'], axis=1)

                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])


            return campione
        elif topic==None and text!=None and Language==None and time != None:
            dfNew=pd.DataFrame() 
            dfNew = corpus.groupby(['cluster', 'tquantile']).size().reset_index(name='Frequencies')

            dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
            dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * sample)))
            

            campione=pd.DataFrame()
            for index, row in dfNew.iterrows():
                tquant = int(row['tquantile'])
                cluster = int(row['cluster'])
                num_samples = int(row['category_sample'])
              

                category_data = corpus.loc[(corpus['cluster'] == cluster) & (corpus['tquantile'] == tquant)].drop(['clear_text', 'tquantile', 'cluster'], axis=1)
                samples = pd.DataFrame(category_data.sample(num_samples))
                campione=pd.concat([campione, samples])


            return campione
        elif text != None and topic==None and Language==None and time== None:

            proportions = corpus['cluster'].value_counts(normalize=True)

            category_samples = round(proportions * sample).astype(int) 


            dfNew=pd.DataFrame() 
            for category, num_samples in category_samples.items():
                category_data = corpus[corpus['cluster'] == category].drop(['clear_text', 'cluster'], axis=1)
                samples = pd.DataFrame(category_data.sample(num_samples))
                dfNew=pd.concat([dfNew, samples])

           




            return dfNew
        elif text == None and topic!=None and Language==None and time== None:

            proportions = corpus[topic].value_counts(normalize=True)
            category_samples = round(proportions * sample).astype(int) 


            dfNew=pd.DataFrame() 
            for category, num_samples in category_samples.items():
                category_data = corpus[corpus[topic] == category]
                samples = pd.DataFrame(category_data.sample(num_samples))
                dfNew=pd.concat([dfNew, samples])




            return dfNew
        
        elif text == None and topic==None and Language!=None and time== None:

            proportions = corpus[Language].value_counts(normalize=True)
            category_samples = round(proportions * sample).astype(int)

            dfNew=pd.DataFrame()
            for category, num_samples in category_samples.items():
                category_data = corpus[corpus[Language] == category]
                samples = pd.DataFrame(category_data.sample(num_samples))
                dfNew=pd.concat([dfNew, samples])

            return dfNew
           

        elif text == None and topic==None and Language==None and time!= None:

            proportions = corpus['tquantile'].value_counts(normalize=True)

            category_samples = round(proportions * sample).astype(int) 


            dfNew=pd.DataFrame() 
            for category, num_samples in category_samples.items():
                category_data = corpus[corpus['tquantile'] == category].drop(['tquantile'], axis=1)
                samples = pd.DataFrame(category_data.sample(num_samples))
                dfNew=pd.concat([dfNew, samples])

            return dfNew
        

        
files = [('facebook Messenger', 'content')]
for el in tqdm(files):


    start_time = time.time()

    data = pd.read_csv(f'Data/{el[0]}.csv', index_col=None)
    print(len(data))
    sampler = TextSampler(data,text=el[1], sample = int(len(data)/4), cluster = 'SOM')
    campione = sampler.Sampler()
    #campione.to_csv(f'{nome_file}.csv')
    end_time = time.time()
    execution_time = end_time - start_time


    print(float(execution_time))

