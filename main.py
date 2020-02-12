import os
import networkx as nx
import joblib
import shutil
import gc
import numpy as np
import random

from config import Config_runner_cora_deepwalk as Config
import NetworkUtilities, ReadLabel, ReadAttribute, ReadHeteGraph, ReadGraph, Evaluation, \
    Read_Write_Embeddings, SimulateWalks
import ganrl

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def filetype_default():
    raise  Exception('unrecognized file type, '
                     'the valid types are one of the following:\n'
                     'edgelist\n'
                     'adjlist\n'
                     'mat\n'
                     'attlist\n'
                     'attmat\n'
                     'attpair\n')


def construct_neighborhood(Config, Gr, neighborhood_type, interdata_path, meta_data_neighborhood, type_word_list):
    # the structure of neighborhood for multiple generators: neighborhood[nei_type_id][repeat_id][node_type][center_id]
    # [nei_id] = weight
    # the structure of neighborhood for one generator: neighborhood[nei_type_id][repeat_id][center_id][nei_id] = weight
    nei_type = neighborhood_type['nei_type']
    center_type = neighborhood_type['center_type']
    node_type = neighborhood_type['node_type']

    if nei_type == 'rw':
        rw_path = interdata_path + 'num_walks_' + str(Config.num_walks) + '/walk_length_' + str(
            Config.walk_length) + '/'
        if not os.path.exists(rw_path):
            os.makedirs(rw_path)
        walk_file = rw_path + 'rw'

        neighbor_file = walk_file + '_neighborhood_ft_'+str(Config.frequency_threshold)
        if os.path.isfile(neighbor_file):
            print('load neighborhood')
            neighborhood = joblib.load(neighbor_file)
        else:
            print('construct neighborhood')
            if os.path.isfile(walk_file):
                print('load random walk')
                walks = SimulateWalks.read_walks_from_file(walk_file)
            else:
                print('construct random walk')
                walks = SimulateWalks.generate_walks_in_RAM(Gr, num_walks=Config.num_walks,
                                                            walk_length=Config.walk_length)

                SimulateWalks.write_walks_to_file(walks, walk_file)
            neighborhood = NetworkUtilities.build_neighborhood_based_walks(walks, Config.num_walks,
                                                                           Config.window_size,
                                                                           Config.num_walks_per_neighborhood,
                                                                           Config.f_remove_neighbors,
                                                                           Config.frequency_threshold,
                                                                           meta_data_neighborhood)
            joblib.dump(neighborhood, neighbor_file)

    elif nei_type == 'metarw':
        rw_path = interdata_path + 'metapath_' + '_'.join(Config.metapath) + '/num_walks_' + str(
            Config.num_walks) + '/walk_length_' + str(Config.walk_length) + '/'
        if not os.path.exists(rw_path):
            os.makedirs(rw_path)
        walk_file = rw_path + 'rw'
        neighbor_file = walk_file + '_neighborhood_ft_'+str(Config.frequency_threshold)
        if os.path.isfile(neighbor_file):
            print('load neighborhood')
            neighborhood = joblib.load(neighbor_file)
        else:
            print('construct neighborhood')
            if os.path.isfile(walk_file):
                print('load random walk')
                walks = SimulateWalks.read_walks_from_file(walk_file)
            else:
                print('construct random walk')
                walks = SimulateWalks.generate_walks_in_RAM(Gr, metapath=Config.metapath,
                                                            num_walks=Config.num_walks,
                                                            walk_length=Config.walk_length)
                SimulateWalks.write_walks_to_file(walks, walk_file)
            neighborhood = NetworkUtilities.build_neighborhood_based_walks(walks, Config.num_walks,
                                                                           Config.window_size,
                                                                           Config.num_walks_per_neighborhood,
                                                                           Config.f_remove_neighbors,
                                                                           Config.frequency_threshold,
                                                                           meta_data_neighborhood)
            joblib.dump(neighborhood, neighbor_file)

    elif nei_type == 'one_hop':
        neighbor_file = interdata_path + Config.f_graph + '_neighborhood_one_hop'
        if os.path.isfile(neighbor_file):
            print('load neighborhood')
            neighborhood = joblib.load(neighbor_file)
        else:
            print('construct neighborhood')
            neighborhood = NetworkUtilities.build_neighborhood_of_one_hop(Gr, meta_data_neighborhood)
            joblib.dump(neighborhood, neighbor_file)

    elif nei_type == 'same_att':
        neighbor_file = interdata_path + 'neighborhood_same_att'
        if os.path.isfile(neighbor_file):
            print('load neighborhood')
            neighborhood = joblib.load(neighbor_file)
        else:
            print('construct neighborhood')
            neighborhood = NetworkUtilities.build_neighborhood_with_same_att(Gr, Config.att_threshold)
            joblib.dump(neighborhood, neighbor_file)

    neighborhood_list = []
    if meta_data_neighborhood != None:
        if center_type == 'all':
            if node_type == 'all':
                neighborhood_list.extend(neighborhood)
            else:
                nei_temp = []
                for nei in neighborhood:
                    nei_temp.append({n_type:nei[n_type]  for n_type in node_type})
                neighborhood_list.extend(nei_temp)
        else:
            center_set = []
            for c_type in center_type:
                center_set.extend(type_word_list[c_type])
            if node_type == 'all':
                nei_temp = []
                for nei in neighborhood:
                    nei_temp.append(
                        {n_type: {c_node: nei[n_type][c_node] for c_node in center_set} for n_type in nei})
                neighborhood_list.extend(nei_temp)
            else:
                nei_temp = []
                for nei in neighborhood:
                    nei_temp.append({n_type: {c_node: nei[n_type][c_node] for c_node in center_set} for n_type in node_type})
                neighborhood_list.extend(nei_temp)
    else:
        neighborhood_list.extend(neighborhood)
    return neighborhood_list


def train(Config,embedding_path, result_path, interdata_path, Gr, debug_data, node_pair_prox_array=None):
    if Config.f_graph == 'HG' and Config.f_GAN == 'sep':
        word_type_map = Gr.graph['vertices_type_map']
        type_word_list = {}
        word_type_set = Gr.graph['vertices_type_set']
        for word_type in word_type_set:
            type_word_list[word_type] = list(Gr.graph[word_type + '_map'].values())
            type_word_list[word_type].sort()
        meta_data_neighborhood = [word_type_map, word_type_set]
        meta_data_GAN = type_word_list
    else:
        meta_data_neighborhood = None
        meta_data_GAN = None

    # neighborhood
    neighborhood_list = []
    center_type_set = []
    node_type_set = []
    for neighborhood_type in Config.f_neighborhood_set:
        neighborhood_list.append(construct_neighborhood(Config, Gr, neighborhood_type, interdata_path, meta_data_neighborhood, meta_data_GAN))
        center_type_set.append(neighborhood_type['center_type'])
        node_type_set.append(neighborhood_type['node_type'])

    gc.collect()

    # NeighborGAN
    if Config.f_initial_random:
        embedding = np.random.rand(len(Gr), Config.embedding_size)
    else:
        embedding = Read_Write_Embeddings.read_embeddings_from_file(embedding_path + 'emb')
    GANRL = ganrl.GANRL(Config, Gr,embedding,result_path,neighborhood_list, center_type_set, node_type_set,meta_data_GAN,debug_data,node_pair_prox_array)
    embed_g, embed_d = GANRL.train()
    Read_Write_Embeddings.write_embeddings_to_file(embed_g, result_path + 'gen' + '-' + Config.tt)
    Read_Write_Embeddings.write_embeddings_to_file(embed_d, result_path + 'dis' + '-' + Config.tt)
    del GANRL,neighborhood_list,meta_data_GAN
    return embed_g, embed_d



def main(Config, G_path, A_path, node_type):
    #'''

    print('reading graph')
    G = ReadGraph.read_edgelist(G_path)

    AG = ReadAttribute.read_attmat(G, A_path)

    HG = ReadHeteGraph.trans_hetegraph(AG, [node_type[0], node_type[1]])

    if Config.f_graph == 'G':
        Gr = G
    elif Config.f_graph == 'AG':
        Gr = AG
        del AG
    elif Config.f_graph == 'HG':
        Gr = HG
        del HG

    gc.collect()

    num_nodes = len(Gr)
    print('number of nodes {}\n'.format(num_nodes))
    print('number of edges {}\n'.format(Gr.number_of_edges()))

    # save some variables for subsequent use
    if Config.flag_classification:
        name_graph_map = Gr.graph[Config.node_type_for_classification + '_map']
        label_dict = ReadLabel.read_label(Config.label_path)  # name/name id  label
        label_list = [(name_graph_map[name], label_dict[name]) for name in label_dict]
        del name_graph_map, label_dict
    else:
        label_list = None

    if Config.flag_link_prediction:
        for test_link_ratio in Config.test_link_ratio_set:
            # generate new network by deleting test and validation links for link prediction
            path_suffix = 'link_prediction/positive_ratio_' + str(Config.positive_ratio) + '/test_link_ratio_' + str(test_link_ratio) + '/'
            result_path_lp = Config.result_path + path_suffix
            interdata_path_lp = Config.interdata_path + path_suffix

            test_link_file = interdata_path_lp + 'test_links'
            print('reading test links')
            test_links = NetworkUtilities.read_test_links(test_link_file)
            print('test links loaded')

            Gr_lp = NetworkUtilities.generate_network_by_deleting_edge(Gr, test_links)

            embedding_path_lp = Config.embedding_path+path_suffix
            gc.collect()
            if Config.f_debug:
                debug_data_lp = [test_links,label_list]
            else:
                debug_data_lp = [None,None]
            embed_g_lp , embed_d_lp = train(Config,embedding_path_lp,result_path_lp,interdata_path_lp, Gr_lp, debug_data_lp)
            Evaluation.evaluation_set(embed_g_lp, result_path_lp, Config.tt, link_samples=test_links,
                                      label_list=label_list, train_ratio_set=Config.train_ratio_set,
                                      classifier=Config.classifier, max_iter=Config.max_iter)
            Evaluation.evaluation_set(embed_d_lp, result_path_lp, Config.tt, link_samples=test_links,
                                      label_list=label_list, train_ratio_set=Config.train_ratio_set,
                                      classifier=Config.classifier, max_iter=Config.max_iter)


    node_pair_prox_array = None
    if label_list==None or Config.flag_convergence_learning:
        prox_array_file = Config.interdata_path + 'proximity_array'
        if os.path.isfile(prox_array_file):
            print('load proximity array')
            node_pair_prox_array = joblib.load(prox_array_file)
        else:
            print('construct proximity array')
            num_random_pairs = 1e7
            node_pair_proximity = []
            sum_common_nei = 0
            sum_common_att = 0
            num_node_pair = 0
            if num_random_pairs < num_nodes * num_nodes:
                pair_set = random.sample(range(num_nodes * num_nodes), int(num_random_pairs))
                for nid in pair_set:
                    i = nid // num_nodes
                    j = nid % num_nodes
                    common_nei = len(set(G.neighbors(i)) & set(G.neighbors(j)))
                    common_att = len(set(Gr.nodes[i]['att_list']) & set(Gr.nodes[j]['att_list']))
                    sum_common_nei += common_nei
                    sum_common_att += common_att
                    node_pair_proximity.append((i, j, (j in G[i]) + 0, common_nei, common_att))
                node_pair_prox_array = np.array(node_pair_proximity)
                avg_common_nei = sum_common_nei / num_random_pairs
                avg_common_att = sum_common_att / num_random_pairs
                node_pair_prox_array[:, 3] = (node_pair_prox_array[:, 3] > avg_common_nei) + 0
                node_pair_prox_array[:, 4] = (node_pair_prox_array[:, 4] > avg_common_att) + 0

            else:
                if Config.directed == False:
                    for i in range(num_nodes):
                        for j in range(i + 1, num_nodes):
                            common_nei = len(set(G.neighbors(i)) & set(G.neighbors(j)))
                            common_att = len(set(Gr.nodes[i]['att_list']) & set(Gr.nodes[j]['att_list']))
                            sum_common_nei += common_nei
                            sum_common_att += common_att
                            num_node_pair += 1
                            node_pair_proximity.append((i, j, (j in G[i]) + 0, common_nei, common_att))
                node_pair_prox_array = np.array(node_pair_proximity)
                avg_common_nei = sum_common_nei / num_node_pair
                avg_common_att = sum_common_att / num_node_pair
                node_pair_prox_array[:, 3] = (node_pair_prox_array[:, 3] > avg_common_nei) + 0
                node_pair_prox_array[:, 4] = (node_pair_prox_array[:, 4] > avg_common_att) + 0
            joblib.dump(node_pair_prox_array, prox_array_file)

    if Config.f_debug:
        debug_data = [None,label_list]
    else:
        debug_data = [None,None]
    embed_g, embed_d = train(Config,Config.embedding_path, Config.result_path, Config.interdata_path, Gr, debug_data, node_pair_prox_array)
    Evaluation.evaluation_set(embed_g, Config.result_path, Config.tt, label_list=label_list,
                              train_ratio_set=Config.train_ratio_set, classifier=Config.classifier,
                              max_iter=Config.max_iter)
    Evaluation.evaluation_set(embed_d, Config.result_path, Config.tt, label_list=label_list,
                              train_ratio_set=Config.train_ratio_set, classifier=Config.classifier,
                              max_iter=Config.max_iter)


def main_run():
    method = 'GANRL'

    print('process initial deepwalk')

    print('process cora')
    loc = locals()

    if not os.path.exists(Config.result_path):
        os.makedirs(Config.result_path)
    if not os.path.exists(Config.interdata_path):
        os.makedirs(Config.interdata_path)

    node_type = ['node', 'attnode']
    main(Config, Config.G_path, Config.A_path, node_type)


if __name__ == "__main__":
    print('process GANRL')
    main_run()