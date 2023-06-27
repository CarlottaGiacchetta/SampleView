import som
import kmeans
import preprocessing
import DOC2VEC
import sampler


class SampleView:
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
    def __init__(self, corpus, sample=1000, text=None, topic=None, Language=None, time=None):
        self.corpus = corpus
        self.text = text
        self.topic=topic
        self.Language = Language
        self.time=time
        self.sample = sample

    


    #FUNZIONE AUSILIARE PER VARIABILE TIME
    def assegna_intervallo(self, valore, intervalli):
        for i, intervallo in enumerate(intervalli):
            if valore >= intervallo[0] and valore <= intervallo[1]:
                return intervallo
        return None



        
    def Sampler(self, vector_size=100, window=5, min_count=1, workers = 1, emb_epochs=10, cluster = 'Kmeans'):
        corpus = self.corpus
        text = self.text
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



            clean = preprocessing.Preprocessing(corpus, text)
            corpus = clean.pulizia()
            model_doc = DOC2VEC.doc2vec(corpus, vector_size, window, min_count, workers, emb_epochs)
            df = model_doc.embedding()
            if cluster=='SOM':
                classe_som=som.SOM(df, corpus)
                corpus = classe_som.cluster_som()



            elif cluster=='Kmeans':
                classe_kmeans = kmeans.Kmeans(df, corpus)
                corpus=classe_kmeans.cluster_KMEANS()


        
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



        model_samp = sampler.SAMPLE(corpus, sample, text, topic, Language, time)
        campione = model_samp.sampling()
        return campione
        

        

 
