import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument



class doc2vec:
    
    def __init__(self, data, vector_size, window, min_count, workers, emb_epochs):   
        

        self.data = data
        self.vector_size=vector_size
        self.window = window
        self.min_count =min_count
        self.epochs=emb_epochs
        self.workers=workers

    def embedding(self): 

        data = self.data

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

