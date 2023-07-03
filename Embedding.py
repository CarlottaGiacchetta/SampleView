from typing import Tuple, Any, Dict

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def doc2vec(data, vector_size: int = 100, window: int = 5, min_count: int = 5, epochs: int = 10, workers: int = 3, **kwargs) -> Tuple[pd.DataFrame, Dict]:

    #trasformo i miei documenti in tagged_docs
    tagged_docs = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(data)]
    # Creazione del modello Doc2vec
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, workers=workers, **kwargs)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    document_vectors = {} 

    for i in range(len(data)):
        document_vectors[i] = model.dv[i]
    df = pd.DataFrame.from_dict(document_vectors, orient='index')
    df = df.apply(lambda x: pd.Series(x), axis=1)
    return df, {'vector_size': vector_size, 'window': window, 'min_count': min_count, 'epochs': epochs, 'workers': workers}


def perform_embedding(data: pd.DataFrame, emb_algo: str,  **kwargs) -> Tuple[Any, dict[str, int]]:
    if emb_algo == 'Doc2Vec':
        doc2_vec_parameters = Doc2Vec.__init__.__code__.co_varnames[:Doc2Vec.__init__.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in doc2_vec_parameters}
        vec, vec_parameters= doc2vec(data, **kwargs)
    elif emb_algo == 'Work2Vec':
        print('Working in Progress :)')
        vec, vec_parameters = None, None
    else:
        raise ValueError("Invalid clustering algorithm")

    return vec, vec_parameters
