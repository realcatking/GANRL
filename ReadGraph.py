'''
Read Graphs from files
'''

'''
The attributes of network
graph
G.graph['map']
    G.graph['node_map']
    G.graph['id_vertices_map']

nodes
G.nodes[]['old_label']

edges
G.edges[]['weight']
'''

import networkx as nx
import networkx.relabel as rlb

'''
default parameters:
weighted = False
directed = False
nodetype = str
'''

def read_edgelist(path, weighted=False, directed=False, nodetype=str):
    '''
    Read Graph from Edge List

    Three formats of edge lists
    1) Node pairs with no data:
    1 2
    2) Python dictionary as data:
    1 2 {'weight':7, 'color':'green'}
    3) Arbitrary data:
    1 2 7 green

    The function declaration:
    read_edgelist(path, comments='#', delimiter=None, create_using=None,
    nodetype=None, data=True, edgetype=None, encoding='utf-8')

    read_weighted_edgelist(path, comments='#', delimiter=None,
    create_using=None, nodetype=None, encoding='utf-8')

    '''

    if directed:
        create_using = nx.DiGraph
    else:
        create_using = nx.Graph

    if weighted:
        data = (('weight',float),)
    else:
        data = True

    str_G = nx.read_edgelist(path, comments="#", create_using=create_using, nodetype=nodetype, data=data)
    int_G = rlb.convert_node_labels_to_integers(str_G, first_label=0, ordering='default', label_attribute='old_label')
    int_G.graph['node_map'] = {label:id for (id, label) in int_G.nodes('old_label')}
    int_G.graph['id_vertices_map'] = {id:label for (id, label) in int_G.nodes('old_label')}
    if not weighted:
        for edge in int_G.edges:
            int_G.edges[edge]['weight'] = 1
    return int_G
