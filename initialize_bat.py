import torch
from relgraph_bat import generate_relation_triplets
import networkx as nx
import torch
import numpy as np
import tqdm
def initialize(target, msg, d_e, d_r, B):

    init_emb_ent = torch.zeros((target.num_ent, d_e)).cuda()
    init_emb_rel = torch.zeros((2*target.num_rel, d_r)).cuda()
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_emb_ent, gain = gain)
    torch.nn.init.xavier_normal_(init_emb_rel, gain = gain)

    # 初始化实体节点图
    G_nx_ent = nx.Graph()
    G_nx_ent_di = nx.DiGraph()
    for j in range(target.num_ent):
        G_nx_ent.add_node(j)
        G_nx_ent_di.add_node(j)
    G_nx_ent.add_edges_from(np.array(msg)[:, 0::2])
    G_nx_ent_di.add_edges_from(np.array(msg)[:, 0::2])
    ent_node_struct_emb = node_feature_init(G_nx_ent)
    init_emb = {'init_emb_ent':init_emb_ent,'ent_node_struct_emb':ent_node_struct_emb}

    relation_triplets, rel_graph, G_nx_rel, rel_node_struct_emb = generate_relation_triplets(msg, target.num_ent, target.num_rel, B)
    rel_graph = rel_graph.to(device=init_emb_ent.device)
    rel_graph.ndata['feat'] = init_emb_rel
    init_rel ={"init_emb_rel":init_emb_rel,'rel_node_struct_emb':rel_node_struct_emb,'A_type':G_nx_rel['A_type']}

    relation_triplets = torch.tensor(relation_triplets).cuda()
    # dis_train = {
    #     node: nx.single_source_dijkstra_path_length(G_nx_ent_di, node)
    #     for node in G_nx_ent_di.nodes
    # }
    G = {'G_nx_ent':G_nx_ent,'G_nx_ent_di':G_nx_ent_di,'G_nx_rel':G_nx_rel['G_nx_rel'],'G_nx_rel_di':G_nx_rel['G_nx_rel_di']}

    return init_emb, init_rel, relation_triplets, rel_graph,G

def normalization(feature):
    a = min(feature)
    b = max(feature)
    c = sum(feature)/len(feature)
    d = np.std(feature)
    feature = [ float((x-c)/(d)) for x in feature]
    return feature

def node_feature_init(G):
    feature_dim = 4
    n = G.number_of_nodes()
    initial_feature = torch.zeros(feature_dim, n).tolist()

    # degree feature
    degree_feature = []
    for idx, node in enumerate(G.nodes()):
        degree_feature.append(G.degree(node))
    degree_feature = normalization(degree_feature)
    initial_feature[0] = degree_feature

    # egonet feature
    # egonet_within = []
    # egonet_without = []
    # n_edges = len(G.edges())
    # for idx, node in enumerate(G.nodes()):
    #     ego_graph = nx.ego_graph(G, node, radius=1)
    #     n_within_edges = len(ego_graph.edges())
    #     n_external_edges = n_edges - n_within_edges
    #     egonet_within.append(n_within_edges)
    #     egonet_without.append(n_external_edges)
    # egonet_within = normalization(egonet_within)
    # egonet_without = normalization(egonet_without)
    # initial_feature[1] = egonet_within
    # initial_feature[2] = egonet_without

    # triangle feature
    triangles = nx.triangles(G)
    triangle_feature = []
    for idx, node in enumerate(G.nodes()):
        triangle_feature.append(triangles[node])
    triangle_feature = normalization(triangle_feature)
    initial_feature[1] = triangle_feature

    # k-core feature
    G.remove_edges_from(nx.selfloop_edges(G))
    kcore = nx.core_number(G)
    kcore_feature = []
    for idx, node in enumerate(G.nodes()):
        kcore_feature.append(kcore[node])
    kcore_feature = normalization(kcore_feature)
    initial_feature[2] = kcore_feature

    # clique feature
    cn = nx.node_clique_number(G)
    clique_feature = []
    for idx, node in enumerate(G.nodes()):
        clique_feature.append(cn[node])
    clique_feature = normalization(clique_feature)
    initial_feature[3] = clique_feature

    initial_feature = torch.Tensor(initial_feature).T
    return initial_feature