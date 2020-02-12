'''
Some utilities about network
'''

import random
import os
import math
import copy
from collections import Counter
import numpy as np
import networkx as nx
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from queue import PriorityQueue as pQueue

_graph = None


def generate_network_by_deleting_edge(G,link_samples):
    New_G = copy.deepcopy(G)
    link_array = np.array(link_samples)
    New_G.remove_edges_from(list(zip(link_array[link_array[:,2]==1, 0], link_array[link_array[:,2]==1, 1])))
    return New_G


def build_neighborhood_based_walks(walks,num_walks,window_size, num_walks_per_neighborhood,f_remove_neighbors,  frequency_threshold = None, meta_data = None):
    if meta_data!=None:
        word_type_map = meta_data[0]
        word_type_set = meta_data[1]
        flag_word_type = True
    else:
        flag_word_type = False
    num_workers = cpu_count()
    num_nodes = len(walks) // num_walks
    num_neighborhood = num_walks // num_walks_per_neighborhood
    n_nei_per_worker = math.ceil(num_neighborhood/num_workers)
    used_workers_nei = math.ceil(num_neighborhood/n_nei_per_worker)
    if used_workers_nei==1:
        if not flag_word_type:
            neighborhood_list=build_neighborhood_mp(walks, num_walks, num_walks_per_neighborhood, window_size,f_remove_neighbors,  frequency_threshold)
        else:
            neighborhood_list=build_neighborhood_hete_mp(walks, num_walks, num_walks_per_neighborhood, window_size, word_type_map,
                                       word_type_set,f_remove_neighbors,  frequency_threshold)
    else:
        n_walks_per_worker = n_nei_per_worker*num_walks_per_neighborhood
        n_walks_per_worker_list = [n_walks_per_worker]*(used_workers_nei-1)+[num_walks-n_walks_per_worker*(used_workers_nei-1)]
        walks_per_worker = []
        for i in range(used_workers_nei-1):
            walks_per_worker.append(walks[i*num_nodes*n_walks_per_worker:(i+1)*num_nodes*n_walks_per_worker])
        walks_per_worker.append(walks[(used_workers_nei-1)*num_nodes*n_walks_per_worker:])
        neighborhood_list = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            if not flag_word_type:
                for neighborhood in executor.map(build_neighborhood_mp, walks_per_worker,n_walks_per_worker_list,[num_walks_per_neighborhood]*used_workers_nei,[window_size]*used_workers_nei,[f_remove_neighbors]*used_workers_nei,  [frequency_threshold]*used_workers_nei):
                    neighborhood_list.extend(neighborhood)

            else:
                for neighborhood in executor.map(build_neighborhood_hete_mp, walks_per_worker,n_walks_per_worker_list,[num_walks_per_neighborhood]*used_workers_nei,[window_size]*used_workers_nei, [word_type_map]*used_workers_nei, [word_type_set]*used_workers_nei,[f_remove_neighbors]*used_workers_nei,  [frequency_threshold]*used_workers_nei):
                    neighborhood_list.extend(neighborhood)

    return neighborhood_list


def build_neighborhood_mp(walks,num_walks,num_walks_per_neighborhood,window_size,f_remove_neighbors,  frequency_threshold):
    neighborhood = []
    num_nodes = len(walks) // num_walks
    num_nei = num_walks//num_walks_per_neighborhood
    for i in range(num_nei):
        flag_first = True
        neighborhood.append({})
        for j in range(num_nodes):
            neighborhood[i][j] = {}
        if i == num_nei - 1:
            n_walks_per_nei = num_walks - num_walks_per_neighborhood * i
        else:
            n_walks_per_nei = num_walks_per_neighborhood
        for k in range(n_walks_per_nei):
            y = i * num_walks_per_neighborhood + k
            for walk in walks[y * num_nodes:(y + 1) * num_nodes]:
                for j in range(len(walk)):
                    start = max(0, j - window_size)
                    end = min(len(walk), j + window_size + 1)
                    for x in range(start, end):
                        if x == j:
                            continue
                        if int(walk[x]) not in neighborhood[i][int(walk[j])]:
                            if f_remove_neighbors == 'intersection':
                                if not flag_first:
                                    continue
                            neighborhood[i][int(walk[j])][int(walk[x])] = 1
                        else:
                            neighborhood[i][int(walk[j])][int(walk[x])] += 1
            if f_remove_neighbors == 'intersection':
                for nod in range(num_nodes):
                    nei_count = copy.deepcopy(neighborhood[i][nod])
                    for nei in nei_count:
                        if neighborhood[i][nod][nei] == k+1:
                            neighborhood[i][nod].pop(nei)
            flag_first = False

        if f_remove_neighbors == 'low_freq':
            for n in range(num_nodes):
                nei_count = copy.deepcopy(neighborhood[i][n])
                for nei in nei_count:
                    if neighborhood[i][n][nei] <= frequency_threshold:
                        neighborhood[i][n].pop(nei)

    return neighborhood


def build_neighborhood_hete_mp(walks,num_walks,num_walks_per_neighborhood,window_size, word_type_map, word_type_set,f_remove_neighbors,  frequency_threshold):
    neighborhood = []
    num_nodes = len(walks) // num_walks
    num_nei = num_walks // num_walks_per_neighborhood
    for i in range(num_nei):
        flag_first = True
        neighborhood.append({})
        for word_type in word_type_set:
            neighborhood[i][word_type] = {}
            for j in range(num_nodes):
                neighborhood[i][word_type][j] = {}
        if i == num_nei - 1:
            n_walks_per_nei = num_walks - num_walks_per_neighborhood * i
        else:
            n_walks_per_nei = num_walks_per_neighborhood
        for k in range(n_walks_per_nei):
            y = i * num_walks_per_neighborhood + k
            for walk in walks[y * num_nodes:(y + 1) * num_nodes]:
                for j in range(len(walk)):
                    start = max(0, j - window_size)
                    end = min(len(walk), j + window_size + 1)
                    for x in range(start, end):
                        if x == j:
                            continue
                        if int(walk[x]) not in neighborhood[i][word_type_map[int(walk[x])]][int(walk[j])]:
                            if f_remove_neighbors == 'intersection':
                                if not flag_first:
                                    continue
                            neighborhood[i][word_type_map[int(walk[x])]][int(walk[j])][int(walk[x])] = 1
                        else:
                            neighborhood[i][word_type_map[int(walk[x])]][int(walk[j])][int(walk[x])] += 1
            if f_remove_neighbors == 'intersection':
                for word_type in word_type_set:
                    for nod in range(num_nodes):
                        nei_count = copy.deepcopy(neighborhood[i][word_type][nod])
                        for nei in nei_count:
                            if neighborhood[i][word_type][nod][nei] == k+1:
                                neighborhood[i][word_type][nod].pop(nei)
            flag_first = False

        if f_remove_neighbors == 'low_freq':
            for word_type in word_type_set:
                for n in range(num_nodes):
                    nei_count = copy.deepcopy(neighborhood[i][word_type][n])
                    for nei in nei_count:
                        if neighborhood[i][word_type][n][nei] <= frequency_threshold:
                            neighborhood[i][word_type][n].pop(nei)

    return neighborhood

def build_neighborhood_of_one_hop(G, meta_data = None):
    if meta_data!=None:
        word_type_map = meta_data[0]
        word_type_set = meta_data[1]

    neighborhood = []
    if meta_data == None:
        neighborhood.append({})
        for n, nbrs in G.adj.items():
            neighborhood[0][n] = {}
            for nbr, attr in nbrs.items():
                neighborhood[0][n][nbr] = attr['weight']
    else:
        neighborhood.append({})
        for word_type in word_type_set:
            neighborhood[0][word_type] = {}
        for n, nbrs in G.adj.items():
            for word_type in word_type_set:
                neighborhood[0][word_type][n] = {}
            for nbr, attr in nbrs.items():
                neighborhood[0][word_type_map[nbr]][n][nbr] = attr['weight']

    return neighborhood

def build_neighborhood_with_same_att(G, threshold = None):
    global _graph
    _graph = G
    num_workers = cpu_count()//2
    num_nodes = len(G)
    node_per_worker = math.ceil(num_nodes / num_workers)
    used_workers = math.ceil(num_nodes / node_per_worker)
    start_node = range(0,num_nodes,node_per_worker)
    end_node = range(node_per_worker,num_nodes+node_per_worker,node_per_worker)
    neighborhood = []
    neighborhood.append({})
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for neighbor in executor.map(build_neighborhood_with_same_att_mp, start_node, end_node,
                                         [threshold] * used_workers,
                                         ):
            neighborhood[0].updatae(neighbor[0])
    return neighborhood

def build_neighborhood_with_same_att_mp(start_node, end_node, threshold = None):
    G = _graph
    num_nodes = len(G)
    neighborhood = []
    neighborhood.append({})
    if end_node > num_nodes:
        end_node=num_nodes
    for n in range(start_node,end_node):
        neighborhood[0][n] = {}
        cnt = 0
        sum_thre = 0
        for nbr, attr in G[n].items():
            neighborhood[0][n][nbr] = attr['weight']
            if threshold == None:
                cnt += 1
                sum_thre += len(set(G.nodes[n]['att_list']) & set(G.nodes[nbr]['att_list']))
        if threshold == None:
            if cnt == 0:
                thre = 1
            else:
                thre = sum_thre / cnt
        else:
            thre = threshold

        for nb in range(num_nodes):
            if nb == n:
                continue
            if len(set(G.nodes[n]['att_list']) & set(G.nodes[nb]['att_list'])) >= thre:
                if nb in neighborhood[0][n]:
                    neighborhood[0][n][nb] += 1
                else:
                    neighborhood[0][n][nb] = 1
    return neighborhood


def read_test_links(file):
    f = open(file, 'r')
    test_links = []
    line = f.readline()
    while line:
        items = line.strip().split()
        line = f.readline()
        test_links.append((int(items[0]), int(items[1]), int(items[2])))
    f.close()
    return test_links
