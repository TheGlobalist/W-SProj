from __future__ import division
import pandas as pd
import networkx as nx
import dt_preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt

def createGraphPerYear(dataset, year):
    insertedWords = set()
    listaAnni = set(dataset['anno'].values)
    grafi = dict()
    for anno in listaAnni:
        print(type(anno))
        datasetTemporale = dataset[dataset['anno'] == anno]
        G=nx.DiGraph()
        for index, row in datasetTemporale.iterrows():
            #Reminder: ogni row è formato da anno, keyword1, keyword2, dizionario utilizzatore keywords - numero volte
            #FASE 1: AGGIUNTA DEI DUE POSSIBILI NODI
            if row.keyword1 not in G:
                G.add_node(row.keyword1)
            if row.keyword2 not in G:
                G.add_node(row.keyword2)
            if not __areNodesConnected(G,row.keyword1, row.keyword2):
                G.add_edge(row.keyword1,row.keyword2)
        grafi[anno] = G
    return grafi


def __areNodesConnected(G, nodeToCheckOne,nodeToCheckTwo):
    return nodeToCheckOne in G.neighbors(nodeToCheckTwo)


dt= dt_preprocessing.loadDatasetOne()
grafiPerAnno = createGraphPerYear(dt,1990)
print(grafiPerAnno[1989])


for x in grafiPerAnno:
    G = grafiPerAnno[x]
    # write edgelist to grid.edgelist
    pr = nx.pagerank(G)
    print(pr)