import pandas as pd


class SAMPLE: 

    
    def __init__(self, data, sample, text, topic, Language, time):


        self.data = data
        self.sample = sample
        self.text = text
        self.topic = topic
        self.Language = Language
        self.time = time


    def sampling(self):
        corpus = self.data

        lista = []
        diz = dict({'topic': self.topic, 'text': self.text, 'Language':self.Language, 'time': self.time})
        diz_final = {}
        for key in list(dict.fromkeys(diz)):
            if diz[key] != None:
                if key == 'text':
                    lista.append('cluster')
                    diz_final[key]='cluster'
                elif key == 'time':
                    lista.append('tquantile')
                    diz_final[key]='tquantile'
                else:
                    lista.append(diz[key])
                    diz_final[key]=diz[key]
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
            
            

            

        return campione, lista
        






