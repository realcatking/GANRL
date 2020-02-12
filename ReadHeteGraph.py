'''
read heterogeneous network or transform attribute network to heterogeneous network
'''

'''
The attributes of heterogeneous network
graph
G.graph['vertices_type_set']
G.graph['map']
    G.graph['paper_map']
    //G.graph['id_paper_map']
    G.graph['author_map']
    //G.graph['id_author_map']
    G.graph['conf_map']
    //G.graph['id_conf_map']
    G.graph['id_name_p']
    G.graph['name_id_p']
    G.graph['id_name_a']
    G.graph['name_id_a']
    G.graph['id_name_c']
    G.graph['name_id_c']
    
    G.graph['node_map']
    G.graph['attnode_map']
    // G.graph['att_map']
    // G.graph['id_att_map']
    
    G.graph['vertices_type_map']
    G.graph['id_vertices_map']

nodes
G.nodes[]['type']
G.nodes[]['old_label']
G.nodes[]['nei_']
    G.nodes[]['nei_author']
    G.nodes[]['nei_paper']
    G.nodes[]['nei_conf']
    
    G.nodes[]['nei_node']
    G.nodes[]['nei_attnode']
    // G.nodes[]['att_list']
    // G.nodes[]['att_value']
    
edges
G.edges[]['type']
G.edges[]['weight']
'''
import copy

def trans_hetegraph(OG,node_type=['node','attnode']):
    G = copy.deepcopy(OG)
    node_num = node_id = G.number_of_nodes()
    node_map_set = []
    if 'vertices_type_set' not in G.graph:
        G.graph['vertices_type_set'] = []
        G.graph['vertices_type_map'] = {}
    if node_type[0]+'_map' not in G.graph:
        G.graph[node_type[0]+'_map'] = G.graph['node_map']
        del G.graph['node_map']
    for node_t in node_type:
        node_map_set.append(node_t+'_map')
        if node_t+'_map' not in G.graph:
            G.graph[node_t+'_map'] = {}
        if node_t not in G.graph['vertices_type_set']:
            G.graph['vertices_type_set'].append(node_t)

    for u,v in G.edges:
        if 'weight' not in G.edges[u,v]:
            G.edges[u,v]['weight'] = 1
        G.edges[u,v]['type'] = node_type[0]+'_'+node_type[0]

    for n in range(node_num):
        G.nodes[n]['type'] = node_type[0]
        G.graph['vertices_type_map'][n] = node_type[0]
        G.nodes[n]['nei_'+node_type[0]] = list(G.adj[n])
        G.nodes[n]['nei_' + node_type[1]] = []
        if 'att_list' in G.nodes[n]:
            for atti in G.nodes[n]['att_list']:
                if G.graph['id_att_map'][atti] not in G.graph[node_map_set[1]]:
                    G.graph[node_map_set[1]][G.graph['id_att_map'][atti]] = node_id
                    G.graph['id_vertices_map'][node_id] = G.graph['id_att_map'][atti]
                    G.add_node(node_id, type=node_type[1], old_label=G.graph['id_att_map'][atti])
                    G.graph['vertices_type_map'][node_id] = node_type[1]
                    node_id += 1

                G.nodes[n]['nei_'+node_type[1]].append(G.graph[node_map_set[1]][G.graph['id_att_map'][atti]])
                G.add_edge(n,G.graph[node_map_set[1]][G.graph['id_att_map'][atti]],
                           type=node_type[0]+'_'+node_type[1],
                           weight=G.nodes[n]['att_value'][G.nodes[n]['att_list'].index(atti)])
                if 'nei_'+node_type[0] not in G.nodes[G.graph[node_map_set[1]][G.graph['id_att_map'][atti]]]:
                    G.nodes[G.graph[node_map_set[1]][G.graph['id_att_map'][atti]]]['nei_'+node_type[0]] = []
                if 'nei_'+node_type[1] not in G.nodes[G.graph[node_map_set[1]][G.graph['id_att_map'][atti]]]:
                    G.nodes[G.graph[node_map_set[1]][G.graph['id_att_map'][atti]]]['nei_'+node_type[1]] = []
                G.nodes[G.graph[node_map_set[1]][G.graph['id_att_map'][atti]]]['nei_'+node_type[0]].append(n)
            del G.nodes[n]['att_list'], G.nodes[n]['att_value']

    del G.graph['att_map'], G.graph['id_att_map']
    return G