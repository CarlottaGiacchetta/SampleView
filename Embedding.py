import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

def doc2vec(data, vector_size, window, min_count, workers, emb_epochs): 


    #trasformo i miei documenti in tagged_docs
    tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(data['content'])]

    # Creazione del modello Doc2vec
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=emb_epochs)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    document_vectors = {} 

    for i in range(len(data)):
        document_vectors[i] = model.dv[i]
    df = pd.DataFrame.from_dict(document_vectors, orient='index')
    df = df.apply(lambda x: pd.Series(x), axis=1)
    return df 


def perform_embedding(data: pd.DataFrame, emb_algo: str, vector_size: int, window: int, min_count: int, workers: int, emb_epochs: int,  **kwargs) -> pd.DataFrame:
    if emb_algo == 'Doc2Vec':
        df = doc2vec(data, vector_size, window, min_count, workers, emb_epochs)
    elif emb_algo == 'Work2Vec':
        print('Working in Progress :)')
    else:
        raise ValueError("Invalid clustering algorithm")

    return df
