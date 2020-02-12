'''
Read Attributes from files

for every node, store
a list of att_id
a list of att_value
'''

'''
The attributes of attribute network
graph
G.graph['att_num']
G.graph['map']
    G.graph['node_map']
    G.graph['id_vertices_map']
    G.graph['att_map']
    G.graph['id_att_map']
    
nodes
G.nodes[]['old_label']
G.nodes[]['att_list']
G.nodes[]['att_value']

edges
G.edges[]['weight']
'''

import networkx as nx
import numpy as np
import networkx.relabel as rlb
import copy


def read_attmat(OG, path):
    G = copy.deepcopy(OG)
    for n in G:
        G.nodes[n]['att_list'] = []
        G.nodes[n]['att_value'] = []
    f = open(path, encoding='UTF-8')
    line = f.readline()
    nodei = 0
    att_num = len(line.strip().split())
    G.graph['att_num'] = att_num
    att_id_map = id_att_map = {i: i for i in range(att_num)}
    while line:
        G.nodes[G.graph['node_map'][str(nodei)]]['att_list'] = []
        G.nodes[G.graph['node_map'][str(nodei)]]['att_value'] = []
        atti = 0
        for items in line.strip().split():
            if items != '0':
                G.nodes[G.graph['node_map'][str(nodei)]]['att_list'].append(atti)
                G.nodes[G.graph['node_map'][str(nodei)]]['att_value'].append(float(items))
            atti += 1
        nodei += 1
        line = f.readline()

    f.close()
    G.graph['att_map'] = att_id_map
    G.graph['id_att_map'] = id_att_map
    return G
