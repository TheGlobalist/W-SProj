from __future__ import division
import pandas as pd
import networkx as nx
import dt_preprocessing
import matplotlib as mpl
from itertools import permutations
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import json
import ast
from sklearn.cluster import AgglomerativeClustering
from LinearThreshold import linear_threshold

def graph_to_edge_matrix(G):
    """Convert a networkx graph into an edge matrix.
    See https://www.wikiwand.com/en/Incidence_matrix for a good explanation on edge matrices
   
    Parameters
    ----------
    G : networkx graph
    """
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1

    return edge_mat

"""def test(G,dsOne, dsTwo):
    for edge in G.edges:
        data = G.get_edge_data(edge[0],edge[1])['relationship']
        autoriPerImportanza = dict()
        for autore in data:
            conteggioAutore = dsTwo[dsTwo['auth1'] == autore]
            print(len(conteggioAutore))
            conteggioAutore2 = dsTwo[dsTwo['auth1'] == autore] if len(conteggioAutore) == 0 else 0
            autoriPerImportanza[edge] = len(conteggioAutore) + len(conteggioAutore2)
        if len(data) > 1:
            permutazioni = permutations(data,2)
            for elemOne, elemTwo in permutazioni:
"""          



def getEdgeWeight(dsTwo, data):
    weight = 0
    intersezione = None
    unione = None
    for autore in data:
        conteggioAutore = dsTwo[dsTwo['auth1'] == autore]
        conteggioAutore2 = dsTwo[dsTwo['auth2'] == autore]
        weight += (len(conteggioAutore) + len(conteggioAutore2)) * data[autore]
    return weight / len(set(dsTwo['auth1']).union(dsTwo['auth2'])) 


def weightedJaccard(dataOne, dataTwo):
    autoriOne = [x for x in dataOne]
    autoriTwo = [x for x in dataTwo]
    return set(autoriOne).intersection(set(autoriTwo)) / set(autoriOne).union(set(autoriTwo))



def createGraphPerYear(anni):
    grafi = dict()
    for anno in anni:
        datasetTemporaleOne = pd.read_csv('./Dataset/'+str(anno)+'/'+str(anno)+'datasetOne.csv')
        datasetTemporaleTwo = pd.read_csv('./Dataset/'+str(anno)+'/'+str(anno)+'datasetTwo.csv')
        G=nx.DiGraph()
        for index, row in datasetTemporaleOne.iterrows():
            #Reminder: ogni row è formato da anno, keyword1, keyword2, dizionario utilizzatore keywords - numero volte
            #FASE 1: AGGIUNTA DEI DUE POSSIBILI NODI
            if row.keyword1 not in G:
                G.add_node(row.keyword1)
            if row.keyword2 not in G:
                G.add_node(row.keyword2)
            if not __areNodesConnected(G,row.keyword1, row.keyword2):
                relazione = ast.literal_eval(row.relationship)
                edgeWeight = getEdgeWeight(datasetTemporaleTwo, relazione)
                G.add_edge(row.keyword1,row.keyword2,relationship=relazione, weight=random.random())
        grafi[anno] = G
        #test(G,datasetTemporaleOne,datasetTemporaleTwo)
    return grafi


def __areNodesConnected(G, nodeToCheckOne,nodeToCheckTwo):
    return nodeToCheckOne in G.neighbors(nodeToCheckTwo)


anni = list(range(2000,2010))
grafiPerAnno = createGraphPerYear(anni)
k = 5
finalTopicsList = dict()

for x in grafiPerAnno:
    G = grafiPerAnno[x]
    pr = nx.pagerank(G)
    #h,a = nx.hits(G,max_iter=10000)
    topKSorted= sorted(pr.items(), key=operator.itemgetter(1))[0:k]
    topKLabels = [x[0] for x in topKSorted]

    test = nx.attr_matrix(G)

    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    #cluster.fit_predict(X)
    topics = []
    for i in range(len(topKLabels)):
        attivazione = linear_threshold(G, [topKLabels[i]])
        topics.append(attivazione)
    #topics = linear_threshold(G, topKLabels)
    topicsAsList = np.array(list(topics.values())).flatten()
    if len(topicsAsList) > 1:
        subgraph = G.subgraph(topicsAsList).copy()
        #clusterized = cluster.fit_predict(nx.adjacency_matrix(subgraph).todense())
        clusterized = nx.clustering(subgraph)
        topicsToInsert = [key for key in clusterized if clusterized[key] > 0.3]
        finalTopicsList[x] = {'graph': G, 'subgraph': subgraph, 'topics': topicsToInsert} 
    else:
        finalTopicsList[x] = {'graph': G, 'subgraph': None, 'topics': topicsAsList} 

topicsFinali = []
for x in grafiPerAnno:
    grafoAnnoCorrente = finalTopicsList[x]
    annoSuccessivo = x+1 if x < 2009 else x
    grafoAnnoSuccessivo = finalTopicsList[annoSuccessivo]
    for topic in grafoAnnoCorrente['topics']:
        if topic in grafoAnnoSuccessivo['topics']:
            chiaveGrafoUno = list(grafoAnnoCorrente['subgraph'][topic]._atlas.keys())[0]
            dataOne = [key for key in grafoAnnoCorrente[topic][chiaveGrafoUno].keys()]
            chiaveGrafoDue = list(grafoAnnoSuccessivo[topic]._atlas.keys())[0]
            dataTwo = [key for key in grafoAnnoSuccessivo[topic][chiaveGrafoDue].keys()]
            if weightedJaccard(dataOne,dataTwo) > 0.04:
                pass
            else:
                print("ciao")

"""
topics
[{0: [...], 1: [...]}, {0: [...], 1: [...]}, {0: [...], 1: [...]}, {0: [...], 1: [...]}, {0: [...], 1: [...]}]
topics[0][0]
['artificial neural network']
sottografoOne = G.subgraph(topics[0][0]).copy()
topics[1][0]
['nonlinear system']
topics[0]
{0: ['artificial neural network'], 1: ['neural network simulation']}
topics[0].values()
dict_values([['artificial neural network'], ['neural network simulation']])
[d['value'] for d in topics[0]]
TypeError: 'int' object is not subscriptable
[topics[0][d] for d in topics[0]]
[['artificial neural network'], ['neural network simulation']]
0: ['artificial neural network']
1: ['neural network simulation']
__len__: 2
np.array([topics[0][d] for d in topics[0]]).flatten()
array(['artificial neural network', 'neural network simulation'],
      dtype='<U25')
np.array([topics[0][d] for d in topics[0]]).flatten()[0]
'artificial neural network'
np.array([topics[0][d] for d in topics[0]]).flatten()[1]
'neural network simulation'
tmp1 = [topics[0][d] for d in topics[0]]
tmp1
[['artificial neural network'], ['neural network simulation']]
sottografoOne = G.subgraph(np.array([topics[0][d] for d in topics[0]]).flatten()).copy()
tmp2 = [topics[1][d] for d in topics[1]]
sottografoTwo = G.subgraph(np.array([topics[1][d] for d in topics[1]]).flatten()).copy()
networkx.exception.NetworkXError: Node ['nonlinear system'] in sequence nbunch is not a valid node.
[topics[1][d] for d in topics[1]]
[['nonlinear system'], ['simulation', 'motion estimation\xa0', 'motion estimation\xa0']]
tmp2
[['nonlinear system'], ['simulation', 'motion estimation\xa0', 'motion estimation\xa0']]
tmp2.flatten()
AttributeError: 'list' object has no attribute 'flatten'
np.array(tmp2).flatten()
array([list(['nonlinear system']),
       list(['simulation', 'motion estimation\xa0', 'motion estimation\xa0'])],
      dtype=object)
np.array(tmp2).flatten()[0]
['nonlinear system']
np.array(tmp2).flatten().flatten()
array([list(['nonlinear system']),
       list(['simulation', 'motion estimation\xa0', 'motion estimation\xa0'])],
      dtype=object)
[item for tmp2 in l for item in tmp2]
NameError: name 'l' is not defined
[item for tmp2 in tmp2 for item in tmp2]
['nonlinear system', 'simulation', 'motion estimation\xa0', 'motion estimation\xa0']
0: 'nonlinear system'
1: 'simulation'
2: 'motion estimation\xa0'
3: 'motion estimation\xa0'
__len__: 4
sottografoTwo = G.subgraph(np.array([item for tmp2 in tmp2 for item in tmp2]).flatten()).copy()
nx.clustering(sottografoOne)
{'artificial neural network': 0, 'neural network simulation': 0}
nx.clustering(sottografoTwo)
{'motion estimation\xa0': 0.5, 'nonlinear system': 0.5, 'simulation': 0.5}
"""