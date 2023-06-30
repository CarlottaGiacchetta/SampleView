import clustering
import preprocessing
import DOC2VEC
import sampler

class SampleView:
    ''' 
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
    - workers -> il numero di elementi di parallelizzazione dell'algoritmo.indica il numero di thread da utilizzare durante l'addestramento del modello.
      Questo parametro specifica il numero di thread paralleli che verranno utilizzati per accelerare il processo di addestramento e l'elaborazione dei documenti.
      Di default 1
    - emb_epochs -> numero di epoche di allenamento dell'algoritmo -> di default 10
    '''
    def __init__(self, corpus, sample=1000, sampling_var = []):
        self.corpus = corpus
        self.text = text
        self.topic=topic
        self.label = label
        self.Language = Language
        self.time=time
        self.sample = sample

        
    def SamplerView(self, vector_size=100, window=5, min_count=1, workers = 1, emb_epochs=10, cluster = 'Kmeans'):
        corpus = self.corpus
        text = self.text
        topic = self.topic
        label = self.label
        Language = self.Language 
        time = self.time
        sample = self.sample
        

        if text!=None:
            model_doc = DOC2VEC.doc2vec(corpus, vector_size, window, min_count, workers, emb_epochs)
            df = model_doc.embedding()
            corpus = clustering.perform_cluster(data: df, cluster_algo: str, **kwargs)

        campione = sampling(corpus, sampling_var)


        return campione

def sampling(corpus, lista):
    dfNew=pd.DataFrame() 
    dfNew = corpus.groupby(lista).size().reset_index(name='Frequencies')
    dfNew['Proportion'] = dfNew['Frequencies'].transform(lambda x: x / x.sum())
    dfNew['category_sample']=dfNew['Proportion'].progress_apply(lambda x: int(round(x * self.sample)))

    campione=pd.DataFrame()
    for index, row in dfNew.iterrows():
        category_data = corpus
        num_samples = int(row['category_sample'])
        for key in diz_final:
            category_data = category_data.loc[(category_data[diz_final[key]] == row[diz_final[key]])]
            samples = pd.DataFrame(category_data.sample(num_samples))
            campione=pd.concat([campione, samples])
    return campione 
