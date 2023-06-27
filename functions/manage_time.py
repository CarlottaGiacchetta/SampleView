class TIME:

    def __init__(self, corpus, time):


        self.corpus = corpus
        self.time = time


    def assegna_intervallo(self, valore, intervalli):
        for i, intervallo in enumerate(intervalli):
            if valore >= intervallo[0] and valore <= intervallo[1]:
                return intervallo
        return None
    

    def create_groups(self):

        min = self.corpus[self.time].min()
        max = self.corpus[self.time].max()
        ampiezza_intervallo = (max - min) / 5

        intervalli = []
        valore_inizio = min
        valore_fine = valore_inizio + ampiezza_intervallo

        for i in range(5):
            intervalli.append((valore_inizio, valore_fine))
            valore_inizio = valore_fine
            valore_fine = valore_inizio + ampiezza_intervallo


        self.corpus['tquantile']=self.corpus[self.time].apply(lambda x: self.assegna_intervallo(x, intervalli))


        return self.corpus

