import TextSampler
import pandas as pd
from tqdm import tqdm
import time
tqdm.pandas()


       
files = [('Facebook', 'content')]
for el in tqdm(files):


    start_time = time.time()

    data = pd.read_csv(f'{el[0]}.csv', index_col=None)
    print(len(data))
    model_sample = TextSampler.SampleView(data,text=el[1], topic = 'score', sample = int(len(data)/4))

    campione, list = model_sample.Sampler(vector_size=100, window=5, min_count=1, workers = 1, emb_epochs=10, cluster = 'Kmeans')
    print(list)
    print(campione)
    print(campione.keys())

    #PROPORZIONE SUL CORPUS
    dfdata = data.groupby(list).size().reset_index(name='Frequencies')

    dfdata['Proportion'] = dfdata['Frequencies'].transform(lambda x: x / x.sum())



    print(dfdata)



    #PROPORZIONE SUL CAMPIONE
    dfs= campione.groupby(list).size().reset_index(name='Frequencies')

    dfs['Proportion'] = dfs['Frequencies'].transform(lambda x: x / x.sum())

    print(dfs)
    if 'cluster' in campione.keys():
        campione = campione.drop('cluster', axis=1)
    #campione.to_csv(f'{nome_file}.csv')
    end_time = time.time()
    execution_time = end_time - start_time


    print(float(execution_time))

    
