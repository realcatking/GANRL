'''
Simulate random walks on a network and return set of walks
'''
import random
import math
import os
from time import time
from six.moves import zip_longest, zip
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

_graph = None
_alias_nodes = None
_alias_edges = None

def simulate_walks_iter(G, metapath = None, num_walks = 1, walk_length = 20, restart_prob = 0):
    nodes = list(G.nodes())
    for cnt in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            if metapath == None:
                yield random_walk(G,node,walk_length, restart_prob)
            else:
                yield random_walk_metapath(G,metapath,node,walk_length,restart_prob)


def _simulate_walks_for_MP(metapath = None, num_walks = 1, walk_length = 20, restart_prob = 0):
    G = _graph
    walks = []
    nodes = list(G.nodes())
    t_0 = time()
    for cnt in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            if metapath == None:
                walks.append(random_walk(G,node,walk_length, restart_prob))
            else:
                if G.nodes[node]['type'] in metapath:
                    walks.append((random_walk_metapath(G,metapath,node,walk_length,restart_prob)))
    print("it took {} seconds to generate {} walks".format(time() - t_0, num_walks))
    return walks


def random_walk(G,start_node=None,walk_length=20,restart_prob=0):
    if not start_node==None:
        walk = [start_node]
    else:
        walk = [random.choice(list(G.nodes()))]
    while len(walk) < walk_length:
        cur = walk[-1]
        if len(G[cur]) > 0:
            if random.random() >= restart_prob:
                walk.append(random.choice(list(G[cur])))
            else:
                walk.append(walk[0])
        else:
            break
    return [str(node) for node in walk]


def random_walk_metapath(G,metapath,start_node=None,walk_length=20,restart_prob=0):
    if not start_node==None:
        walk = [start_node]
    else:
        start_node = random.choice(list(G.nodes()))
        while G.nodes[start_node]['type'] not in metapath:
            start_node = random.choice(list(G.nodes()))
        walk = [start_node]

    while len(walk) < walk_length:
        cur = walk[-1]
        nei_type = metapath[metapath.index(G.nodes[cur]['type'])+1]
        if len(G.nodes[cur]['nei_'+nei_type]) > 0:
            if random.random() >= restart_prob:
                walk.append(random.choice(G.nodes[cur]['nei_'+nei_type]))
            else:
                walk.append(walk[0])
        else:
            break
    return [str(node) for node in walk]


def list_even_split(n, iterable, padvalue = None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def generate_walks_in_RAM(G, metapath = None, num_walks = 1, walk_length = 20,
                          restart_prob = 0, num_workers = cpu_count()):
    global _graph
    _graph = G
    walks_list = []

    if num_walks <= num_workers:
        walks_per_worker = [1 for x in range(num_walks)]
    else:
        walks_per_worker = [len(list(filter(lambda z: z!=None, [y for y in x])))
                            for x in list_even_split(math.ceil(num_walks/num_workers),range(1,num_walks+1))]

    real_num_workers = len(walks_per_worker)

    with ProcessPoolExecutor(max_workers = num_workers) as executor:
        for walks in executor.map(_simulate_walks_for_MP, [metapath]*real_num_workers,
                                  walks_per_worker, [walk_length]*real_num_workers, [restart_prob]*real_num_workers):
            walks_list.extend(walks)

    return walks_list


def write_walks_to_file(walks, walk_file):
    with open(walk_file, 'w') as fout:
        for walk in walks:
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))


def read_walks_from_file(walk_file):
    walks = []
    with open(walk_file, 'r') as fin:
        line = fin.readline()
        while line:
            walks.append(line.strip().split())
            line = fin.readline()
    return walks
