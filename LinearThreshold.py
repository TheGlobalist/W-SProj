import networkx as nx
import random
import copy

def linear_threshold(G,S):
    for n in nx.nodes(G):
        G.node[n]['threshold'] = random.random()
    wave = 0
    diffusion = {}
    active = set() #cambia a set in quanto oneroso
    while True:
        added = []
        wave += 1
        for n in nx.nodes(G):
            if n not in active and n in S:
                influence = 0
                for edge in G.in_edges(n,data=True):
                    if edge[0] not in active:
                        try :
                            influence += 1/ G.in_degree(n)
                        except ZeroDivisionError:
                            continue
                        if influence >= G.node[n]['threshold']:
                            active.add(n)
                            added.append(n)
        if len(added) == 0:
            break
        else:
            diffusion[wave] = added
    return diffusion