import pandas as pd
import networkx as nx

def createGraphPerYear(dataset, year):
    insertedWords = set()
    listaAnni = set(dataset['Anno'].values)
    grafi = dict()
    for anno in listaAnni:
        datasetTemporale = dataset[dataset['Anno'] == anno]
        G=nx.Graph()
        for index, row in datasetTemporale.iterrows():
            #Reminder: ogni row è formato da anno, keyword1, keyword2, dizionario utilizzatore keywords - numero volte
            
            #FASE 1: AGGIUNTA DEI DUE POSSIBILI NODI
            if row.keyword1 not in G:
                G.add_node(row.keyword1)
            if row.keyword2 not in G:
                G.add_node(row.keyword2)
        grafi[anno] = G
    pass
