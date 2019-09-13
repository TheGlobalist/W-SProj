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
import re, math 
from collections import Counter 
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





def getEdgeWeight(dsTwo, data):
    weight = 0
    for autore in data:
        conteggioAutore = dsTwo[dsTwo['auth1'] == autore]
        conteggioAutore2 = dsTwo[dsTwo['auth2'] == autore]
        weight += (len(conteggioAutore) + len(conteggioAutore2)) * data[autore]
    return weight / len(set(dsTwo['auth1']).union(dsTwo['auth2'])) 


def weightedJaccard(dataOne, dataTwo):
    autoriOne = [x for x in dataOne]
    autoriTwo = [x for x in dataTwo]
    return len(set(autoriOne).intersection(set(autoriTwo))) / len(set(autoriOne).union(set(autoriTwo)))


  
WORD = re.compile(r'\w+') 
def get_cosine(vec1, vec2): 
    intersection = set(vec1.keys()) & set(vec2.keys()) 
    numerator = sum([vec1[x] * vec2[x] for x in intersection]) 
    sum1 = sum([vec1[x]**2 for x in vec1.keys()]) 
    sum2 = sum([vec2[x]**2 for x in vec2.keys()]) 
    denominator = math.sqrt(sum1) * math.sqrt(sum2) 
    if not denominator: 
        return 0.0 
    else: 
        return float(numerator) / denominator 

def text_to_vector(text): 
    words = WORD.findall(text) 
    return Counter(words) 

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
                G.add_edge(row.keyword1,row.keyword2,relationship=relazione, weight=edgeWeight)
        grafi[anno] = G
        #test(G,datasetTemporaleOne,datasetTemporaleTwo)
    return grafi


def __areNodesConnected(G, nodeToCheckOne,nodeToCheckTwo):
    return nodeToCheckOne in G.neighbors(nodeToCheckTwo)


anni = list(range(2000,2019))
grafiPerAnno = createGraphPerYear(anni)
k = 5
finalTopicsList = dict()

for x in grafiPerAnno:
    G = grafiPerAnno[x]
    pr = nx.pagerank(G)
    #h,a = nx.hits(G,max_iter=10000)
    topKSorted= sorted(pr.items(), key=operator.itemgetter(1))[0:k]
    topKLabels = [x[0] for x in topKSorted]


    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    #test = cluster.fit_predict(X.toarray())
    topics = []
    for i in range(len(topKLabels)):
        attivazione = linear_threshold(G, [topKLabels[i]])
        topics.append(attivazione)
    topicsAsList = []
    for dizionario in topics:
        for chiave in dizionario:
            topicsAsList.extend(dizionario[chiave])
    topicsAsList = set(topicsAsList)
    if len(topicsAsList) > 1:
        subgraph = G.subgraph(topicsAsList).copy()
        clusterized = cluster.fit_predict(nx.adjacency_matrix(subgraph).todense())
        topicsToInsert = [list(topicsAsList)[i] for i in range(len(clusterized)) if clusterized[i] == 1]
        finalTopicsList[x] = {'graph': G, 'subgraph': subgraph, 'topics': topicsToInsert} 
    else:
        finalTopicsList[x] = {'graph': G, 'subgraph': None, 'topics': topicsAsList} 

topicsFinali = []
for x in grafiPerAnno:
    if x == 2018:
        break
    grafoAnnoCorrente = finalTopicsList[x]
    annoSuccessivo = x+1 if x < 2018 else x
    grafoAnnoSuccessivo = finalTopicsList[annoSuccessivo]
    topicOne = grafoAnnoCorrente['topics']
    topicTwo =  set(grafoAnnoSuccessivo['topics'])
    print("The year is... " + str(x))
    print("-------------------------------")
    print("Checking\n" + str(x) + "'s topics: " + str(topicOne) + "\n" + str(x+1) + "'s topics: " + str(topicTwo))
    for argomento in topicOne:
        print("Now checking keyword..." + str(argomento))
        if grafoAnnoSuccessivo['subgraph'].has_node(argomento):
            print(str(argomento) + " is inside the list! let's see the topic similarity...")
            similarity = weightedJaccard(grafoAnnoCorrente['topics'], grafoAnnoSuccessivo['topics'])
            vecOne = text_to_vector(', '.join(topicOne))
            vecTwo = text_to_vector(', '.join(topicTwo))
            cosine = get_cosine(vecOne,vecTwo)
            if cosine > 0.3:
                print("Threshold exceeded! I can merge the topics...")
                topicsFinali.extend(set(topicOne).union(topicTwo))
print(topicsFinali)
