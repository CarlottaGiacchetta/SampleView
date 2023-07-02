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
    model_sample = TextSampler.SampleView(data, text = el[1], sampling_var=['score'], sample = int(len(data)/4))

    campione, sampling_var = model_sample.SamplerView(vector_size=100, window=5, min_count=1, workers = 1, emb_epochs=10, cluster = 'kmeans')
    print(sampling_var)
    print(campione)
    print(campione.keys())

    #PROPORZIONE SUL CORPUS
    dfdata = data.groupby(sampling_var).size().reset_index(name='Frequencies')

    dfdata['Proportion'] = dfdata['Frequencies'].transform(lambda x: x / x.sum())



    print(dfdata)



    #PROPORZIONE SUL CAMPIONE
    dfs= campione.groupby(sampling_var).size().reset_index(name='Frequencies')

    dfs['Proportion'] = dfs['Frequencies'].transform(lambda x: x / x.sum())

    print(dfs)
    if 'cluster' in campione.keys():
        campione = campione.drop('cluster', axis=1)
    #campione.to_csv(f'{nome_file}.csv')
    end_time = time.time()
    execution_time = end_time - start_time


    print(float(execution_time))

    
