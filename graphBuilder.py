import pandas as pd
import networkx as nx

def createGraphPerYear(dataset, year):
    insertedWords = set()
    G=nx.Graph()
    for index, row in dataset.iterrows():
    pass