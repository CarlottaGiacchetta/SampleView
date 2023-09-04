from typing import Tuple, Any, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument

from transformers import BertTokenizer, BertModel
import torch


def BERT_model(data: pd.Series, model_name: str) -> Tuple[pd.DataFrame, Dict]:
    print('dddddddddddddddddddddddd')
    # Carica il tokenizer di BERT
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Carica il modello di BERT
    model = BertModel.from_pretrained(model_name)

    # Inizializza una lista per i vettori di output
    document_vectors = list()
    # Itera sui documenti
    for document in tqdm(data, desc='GET VECTORS'):
        # Tokenizzazione del documento
        tokens = tokenizer.encode(document, add_special_tokens=True)

        # Converti i token in tensori PyTorch
        input_ids = torch.tensor(tokens).unsqueeze(0)

        # Attiva il modello di BERT in modalità valutazione
        model.eval()

        # Ottieni l'output del modello di BERT
        with torch.no_grad():
            outputs = model(input_ids)

        # Estrai l'ultimo strato nascosto (output[0]) dal modello di BERT
        last_hidden_state = outputs[0]

        # Calcola la media dei vettori sull'asse 1 (dimensione dei token)
        document_vector = torch.mean(last_hidden_state, dim=1)

        # Aggiungi il vettore del documento alla lista
        document_vectors.append(document_vector.flatten().tolist())
        #document_vectors[document] = document_vector.flatten().tolist()

    #df = pd.DataFrame({'vec': document_vectors})
    df = pd.DataFrame(document_vectors, index=range(0, len(document_vectors)))
    #df = pd.DataFrame.from_dict(document_vectors, orient='index')

    return df, {'model': model_name}


def doc2vec(data: pd.Series, vector_size: int = 100, window: int = 5, min_count: int = 5, epochs: int = 10, workers: int = 3, **kwargs) -> Tuple[pd.DataFrame, Dict]:

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
    return df, {'model': 'Doc2Vec', 'vector_size': vector_size, 'window': window, 'min_count': min_count, 'epochs': epochs}


def word2vec(data: pd.Series, vector_size: int = 100, window: int = 5, min_count: int = 5, epochs: int = 10, workers: int = 3,
             **kwargs):

    embeddings = Word2Vec(sentences=data,
                          vector_size=vector_size,
                          window=window,
                          min_count=min_count,
                          epochs=epochs, workers=workers, **kwargs)
    document_vectors = list()
    for doc in data:
        vec = np.array([embeddings.wv[x] for x in doc.split()])
        document_vectors.append(np.mean(vec, axis=0))

    df = pd.DataFrame({'vec': document_vectors})
    # df = pd.DataFrame(document_vectors, index=range(0, len(document_vectors)))

    return df, {'model': 'Word2Vec', 'vector_size': vector_size, 'window': window, 'min_count': min_count,
                'epochs': epochs}


def perform_embedding(data: pd.Series, emb_algo: str,  **kwargs) -> Tuple[Any, dict[str, int]]:
    supported_model = ('Doc2Vec', 'Word2Vec', 'bert-base-uncased', 'bert-large-uncased', 'distilbert-base-uncased')
    if emb_algo == 'Doc2Vec':
        doc2_parameters = Doc2Vec.__init__.__code__.co_varnames[:Doc2Vec.__init__.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in doc2_parameters}
        vec, vec_parameters= doc2vec(data, **kwargs)
    elif emb_algo == 'Word2Vec':
        word2vec_parameters = Word2Vec.__init__.__code__.co_varnames[:Word2Vec.__init__.__code__.co_argcount]
        kwargs = {key: value for key, value in kwargs.items() if key in word2vec_parameters}
        vec, vec_parameters= doc2vec(data, **kwargs)
    elif emb_algo in ['bert-base-uncased', 'bert-large-uncased', 'distilbert-base-uncased']:
        vec, vec_parameters = BERT_model(data, emb_algo)
    else:
        raise ValueError(f"Invalid Embeddings algorithm, try with {', '.join(supported_model)}")

    return vec, vec_parameters
