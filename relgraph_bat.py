import dgl
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import *
import numpy as np
import math
import igraph
import torch
import networkx as nx

def create_relation_graph(triplet, num_ent, num_rel):
	ind_h = triplet[:,:2]
	ind_t = triplet[:,1:]
	

	E_h = csr_matrix((np.ones(len(ind_h)), (ind_h[:, 0], ind_h[:, 1])), shape=(num_ent, 2 * num_rel))
	E_t = csr_matrix((np.ones(len(ind_t)), (ind_t[:, 1], ind_t[:, 0])), shape=(num_ent, 2 * num_rel))

	diag_vals_h = E_h.sum(axis=1).A1
	diag_vals_h[diag_vals_h!=0] = 1/(diag_vals_h[diag_vals_h!=0]**2)

	diag_vals_t = E_t.sum(axis=1).A1
	diag_vals_t[diag_vals_t!=0] = 1/(diag_vals_t[diag_vals_t!=0]**2)


	D_h_inv = csr_matrix((diag_vals_h, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
	D_t_inv = csr_matrix((diag_vals_t, (np.arange(num_ent), np.arange(num_ent))), shape=(num_ent, num_ent))
	D_inv = D_h_inv+D_t_inv


	A_h = E_h.transpose() @ D_inv @ E_h
	A_t = E_t.transpose() @ D_inv @ E_t
	A_ht = E_h.transpose() @ D_inv @ E_t
	A_th = E_t.transpose() @ D_inv @ E_h
	# A = {'h2h':A_h,'t2t':A_t,'h2t':A_ht,'t2h':A_th}
	Tensor_Ah2h = torch.from_numpy(A_h.toarray()).unsqueeze(dim=-1).to(torch.float32)
	Tensor_At2t = torch.from_numpy(A_t.toarray()).unsqueeze(dim=-1).to(torch.float32)
	Tensor_Ah2t = torch.from_numpy(A_ht.toarray()).unsqueeze(dim=-1).to(torch.float32)
	Tensor_At2h = torch.from_numpy(A_th.toarray()).unsqueeze(dim=-1).to(torch.float32)
	A = torch.cat([Tensor_Ah2h, Tensor_At2t, Tensor_Ah2t, Tensor_At2h], dim=-1)
	A[A > 0] = 1


	return A_h + A_t + A_ht + A_th, A

def get_relation_triplets(G_rel, B):
	rel_triplets = []
	for tup in G_rel.get_edgelist():
		h,t = tup
		tupid = G_rel.get_eid(h,t)
		w = G_rel.es[tupid]["weight"]
		rel_triplets.append((int(h), int(t), float(w)))
	rel_triplets = np.array(rel_triplets)

	nnz = len(rel_triplets)

	temp = (-rel_triplets[:,2]).argsort()
	weight_ranks = np.empty_like(temp)
	weight_ranks[temp] = np.arange(nnz) + 1

	relation_triplets = []
	rel_weight =[]
	for idx,triplet in enumerate(rel_triplets):
		h,t,w = triplet
		rk = int(math.ceil(weight_ranks[idx]/nnz*B))-1
		relation_triplets.append([int(h), int(t), rk])
		rel_weight.append(w)

		assert rk >= 0
		assert rk < B
	
	return np.array(relation_triplets),rel_weight

def generate_relation_triplets(triplet, num_ent, num_rel, B):
	A, A_type = create_relation_graph(triplet, num_ent, num_rel)
	G_rel = igraph.Graph.Weighted_Adjacency(A)
	relation_triplets,rel_weight = get_relation_triplets(G_rel, B)

	G_nx = nx.Graph()
	G_nx_di = nx.DiGraph()
	for j in range(num_rel):
		G_nx.add_node(j)
		G_nx_di.add_node(j)
	G_nx.add_edges_from(relation_triplets[:, :2])
	G_nx_di.add_edges_from(relation_triplets[:,:2])
	node_feat = node_feature_init(G_nx)

	rel_graph = dgl.graph((relation_triplets[...,0],relation_triplets[...,1]))
	rel_graph.edata['weight'] = torch.tensor(rel_weight)
	G = {'A_type':A_type,'G_nx_rel':G_nx,'G_nx_rel_di':G_nx_di}

	return relation_triplets,rel_graph,G,node_feat

def normalization(feature):
    a = min(feature)
    b = max(feature)
    c = sum(feature)/len(feature)
    d = np.std(feature)
    feature = [ float((x-c)/(d)) for x in feature]
    return feature

def node_feature_init(G):
    feature_dim = 3
    n = G.number_of_nodes()
    initial_feature = torch.zeros(feature_dim, n).tolist()

    # degree feature
    # degree_feature = []
    # for idx, node in enumerate(G.nodes()):
    #     degree_feature.append(G.degree(node))
    # degree_feature = normalization(degree_feature)
    # initial_feature[0] = degree_feature

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

    # # triangle feature
    # triangles = nx.triangles(G)
    # triangle_feature = []
    # for idx, node in enumerate(G.nodes()):
    #     triangle_feature.append(triangles[node])
    # triangle_feature = normalization(triangle_feature)
    # initial_feature[1] = triangle_feature
	#
    # # k-core feature
    # G.remove_edges_from(nx.selfloop_edges(G))
    # kcore = nx.core_number(G)
    # kcore_feature = []
    # for idx, node in enumerate(G.nodes()):
    #     kcore_feature.append(kcore[node])
    # kcore_feature = normalization(kcore_feature)
    # initial_feature[2] = kcore_feature

    # clique feature
    # cn = nx.node_clique_number(G)
    # clique_feature = []
    # for idx, node in enumerate(G.nodes()):
    #     clique_feature.append(cn[node])
    # clique_feature = normalization(clique_feature)
    # initial_feature[3] = clique_feature

    initial_feature = torch.Tensor(initial_feature).T
    return initial_feature