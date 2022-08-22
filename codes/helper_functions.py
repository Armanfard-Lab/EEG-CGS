import torch
from torch import nn
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import dgl

def createIdx(n):
    idx = []
    for i in range(n+1):
        idx.append(i)
    return(idx)

def avg_clips(final_adj, final_features, nb_nodes, div_train):
    list_adj = [final_adj[x::nb_nodes] for x in range(0, nb_nodes)]
    list_features = [final_features[x::nb_nodes] for x in range(0, nb_nodes)]
    print('Averaging graphs/features across {} adj_mat samples...'.format(int(len(list_adj[0]) / div_train)))
    save_adj = []
    save_features = []
    for l in range(0, len(list_adj)):
        for m in range(0,len(list_adj[0]),int(len(list_adj[0])/div_train)):
            mean_adj = torch.mean(list_adj[l][m:m+int(len(list_adj[0])/div_train)],0)
            mean_features = torch.mean(list_features[l][m:m+int(len(list_features[0])/div_train)],0)
            save_adj.append(mean_adj.detach())
            save_features.append(mean_features)
    save_adj = torch.stack(save_adj)
    len_adj = len(save_adj)
    save_features = torch.stack(save_features)
    list_adj = [save_adj[x::div_train] for x in range(0, div_train)]
    list_features = [save_features[x::div_train] for x in range(0, div_train)]
    list_features = torch.cat(list_features, 0)
    print('Averaging DONE!')
    print('------------------------------')

    return list_adj, list_features, len_adj

def adj_concatenate(list_adj, len_adj, nb_nodes, idex):
    adj =[]
    for idx, lst in enumerate(list_adj):
        _adj = nn.ConstantPad1d((nb_nodes * idex[idx], len_adj - nb_nodes * (idex[idx] + 1)), 0)(lst)
        # print(_adj.shape)
        adj.append(_adj.detach())
    adj = torch.cat(adj, 0)
    return adj

def make_dgl_graph(adj):
    # Convert adj to dgl graph"
    nx_graph = nx.from_scipy_sparse_matrix(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)

    return dgl_graph

def sz_set(final_adj, final_features, ano_label, seizure_label, nb_nodes, div_sz, div_nonsz):
    sz = (seizure_label != 0).nonzero()
    nonsz = (seizure_label == 0).nonzero()
    # sz set
    sz_adj = [final_adj[t.detach().item()] for t in sz]
    sz_feature = [final_features[t.detach().item()] for t in sz]
    sz_label = [ano_label[t.detach().item()] for t in sz]
    sz_adj = torch.cat(sz_adj, 0)
    sz_feature = torch.cat(sz_feature, 0)
    sz_label = torch.cat(sz_label, 0)
    list_sz_adj = [sz_adj[x::nb_nodes] for x in range(0, nb_nodes)]
    list_sz_feature = [sz_feature[x::nb_nodes] for x in range(0, nb_nodes)]
    list_sz_label = [sz_label[x::nb_nodes] for x in range(0, nb_nodes)]
    save_sz_adj = []
    save_sz_label = []
    save_sz_feature = []
    for l in range(0, len(list_sz_adj)):
        for m in range(0, len(list_sz_adj[0]), int(len(list_sz_adj[0]) / div_sz)):
            mean_adj = torch.mean(list_sz_adj[l][m:m + int(len(list_sz_adj[0]) / div_sz)], 0)
            mean_label = torch.mean(list_sz_label[l][m:m + int(len(list_sz_label[0]) / div_sz)], 0)
            mean_feature = torch.mean(list_sz_feature[l][m:m + int(len(list_sz_feature[0]) / div_sz)], 0)
            save_sz_adj.append(mean_adj.detach())
            save_sz_label.append(mean_label)
            save_sz_feature.append(mean_feature)

    # save_sz_adj = torch.stack(save_sz_adj)
    save_sz_feature = torch.stack(save_sz_feature)
    save_sz_label = torch.stack(save_sz_label)
    list_sz_adj = [save_sz_adj[x::div_sz] for x in range(0, div_sz)]
    list_sz_feature = [save_sz_feature[x::div_sz] for x in range(0, div_sz)]
    list_sz_feature = torch.cat(list_sz_feature, 0)
    list_sz_label = [save_sz_label[x::div_sz] for x in range(0, div_sz)]
    list_sz_label = torch.cat(list_sz_label, 0)


    # nosz set
    nonsz_adj = [final_adj[t.detach().item()] for t in nonsz]
    nonsz_feature = [final_features[t.detach().item()] for t in nonsz]
    nonsz_label = [ano_label[t.detach().item()] for t in nonsz]
    nonsz_adj = torch.cat(nonsz_adj, 0)
    nonsz_feature = torch.cat(nonsz_feature, 0)
    nonsz_label = torch.cat(nonsz_label, 0)
    list_nonsz_adj = [nonsz_adj[x::nb_nodes] for x in range(0, nb_nodes)]
    list_nonsz_feature = [nonsz_feature[x::nb_nodes] for x in range(0, nb_nodes)]
    list_nonsz_label = [nonsz_label[x::nb_nodes] for x in range(0, nb_nodes)]
    save_nonsz_adj = []
    save_nonsz_label = []
    save_nonsz_feature = []
    for l in range(0, len(list_nonsz_adj)):
        for m in range(0, len(list_nonsz_adj[0]), int(len(list_nonsz_adj[0]) / div_nonsz)):
            mean_adj = torch.mean(list_nonsz_adj[l][m:m + int(len(list_nonsz_adj[0]) / div_nonsz)], 0)
            mean_label = torch.mean(list_nonsz_label[l][m:m + int(len(list_nonsz_label[0]) / div_nonsz)], 0)
            mean_feature = torch.mean(list_nonsz_feature[l][m:m + int(len(list_nonsz_feature[0]) / div_nonsz)], 0)
            save_nonsz_adj.append(mean_adj.detach())
            save_nonsz_label.append(mean_label)
            save_nonsz_feature.append(mean_feature)

    # save_nonsz_adj = torch.stack(save_nonsz_adj)
    save_nonsz_feature = torch.stack(save_nonsz_feature)
    save_nonsz_label = torch.stack(save_nonsz_label)
    list_nonsz_adj = [save_nonsz_adj[x::div_nonsz] for x in range(0, div_nonsz)]
    list_nonsz_feature = [save_nonsz_feature[x::div_nonsz] for x in range(0, div_nonsz)]
    list_nonsz_feature = torch.cat(list_nonsz_feature, 0)
    list_nonsz_label = [save_nonsz_label[x::div_nonsz] for x in range(0, div_nonsz)]
    list_nonsz_label = torch.cat(list_nonsz_label, 0)

    idex = createIdx(int(len(save_sz_adj + save_nonsz_adj) / nb_nodes) - 1)
    list_adj = list_sz_adj + list_nonsz_adj
    features = torch.cat((list_sz_feature, list_nonsz_feature), 0)
    ano_label = torch.cat((list_sz_label, list_nonsz_label), 0)

    adj = []
    for idx, lst in enumerate(list_adj):
        _adj = nn.ConstantPad1d((nb_nodes * idex[idx], len(save_sz_adj + save_nonsz_adj)
                                 - nb_nodes * (idex[idx] + 1)), 0)(torch.stack(lst))
        # print(_adj.shape)
        adj.append(_adj.detach())
    adj = torch.cat(adj, 0)

    sz_label = np.ones(len(save_sz_adj))
    nonsz_label = np.zeros(len(save_nonsz_adj))
    save_seizure_label = np.concatenate((sz_label, nonsz_label), axis=0)

    return adj, features, ano_label, save_seizure_label


def sz_set_small(final_adj, final_features, ano_label, seizure_label, nb_nodes, div_sz):
    sz = (seizure_label != 0).nonzero()
    # sz set
    sz_adj = [final_adj[t.detach().item()] for t in sz]
    sz_feature = [final_features[t.detach().item()] for t in sz]
    sz_label = [ano_label[t.detach().item()] for t in sz]
    sz_adj = torch.cat(sz_adj, 0)
    sz_feature = torch.cat(sz_feature, 0)
    sz_label = torch.cat(sz_label, 0)
    list_sz_adj = [sz_adj[x::nb_nodes] for x in range(0, nb_nodes)]
    list_sz_feature = [sz_feature[x::nb_nodes] for x in range(0, nb_nodes)]
    list_sz_label = [sz_label[x::nb_nodes] for x in range(0, nb_nodes)]
    save_sz_adj = []
    save_sz_label = []
    save_sz_feature = []
    for l in range(0, len(list_sz_adj)):
        for m in range(0, len(list_sz_adj[0]), int(len(list_sz_adj[0]) / div_sz)):
            mean_adj = torch.mean(list_sz_adj[l][m:m + int(len(list_sz_adj[0]) / div_sz)], 0)
            mean_label = torch.mean(list_sz_label[l][m:m + int(len(list_sz_label[0]) / div_sz)], 0)
            mean_feature = torch.mean(list_sz_feature[l][m:m + int(len(list_sz_feature[0]) / div_sz)], 0)
            save_sz_adj.append(mean_adj.detach())
            save_sz_label.append(mean_label)
            save_sz_feature.append(mean_feature)

    # save_sz_adj = torch.stack(save_sz_adj)
    save_sz_feature = torch.stack(save_sz_feature)
    save_sz_label = torch.stack(save_sz_label)
    list_sz_adj = [save_sz_adj[x::div_sz] for x in range(0, div_sz)]
    list_sz_feature = [save_sz_feature[x::div_sz] for x in range(0, div_sz)]
    list_sz_label = [save_sz_label[x::div_sz] for x in range(0, div_sz)]


    idex = createIdx(int(len(save_sz_adj) / nb_nodes) - 1)
    list_adj = list_sz_adj
    features = torch.cat((list_sz_feature), 0)
    ano_label = torch.cat((list_sz_label), 0)

    adj = []
    for idx, lst in enumerate(list_adj):
        _adj = nn.ConstantPad1d((nb_nodes * idex[idx], len(save_sz_adj)
                                 - nb_nodes * (idex[idx] + 1)), 0)(torch.stack(lst))
        # print(_adj.shape)
        adj.append(_adj.detach())
    adj = torch.cat(adj, 0)

    return adj, features, ano_label
