import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases

class Preprocessing:


    def __init__(self, data, text):        
        self.data = data
        self.text = text

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
        

    def pulizia(self):

        data = self.data
        text = self.text

        lemmatizer = WordNetLemmatizer()
        data['clear_text'] = data[text].apply(lambda x: self.clear_text(x))
        data['clear_text'] = data['clear_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
        bigram_model = Phrases(data['clear_text'],  min_count=5, threshold=0.2)
        data['clear_text'] = data['clear_text'].apply(lambda x: bigram_model[bigram_model[x]])
        
        return data

    

    