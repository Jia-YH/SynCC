import numpy as np
import networkx as nx
from torch_scatter import scatter

import dgl
import torch


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def eigenvector_centrality(g):
    graph = dgl.to_networkx(g)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(g.nodes().shape[0])]
    return torch.tensor(x, dtype=torch.float32).to(g.device)


def compute_pr(g, damp: float = 0.85, k: int = 10):
    num_nodes = g.nodes().shape[0]
    deg_out = g.out_degrees()
    x = torch.ones((num_nodes, )).to(g.device).to(torch.float32)
    edge_index = g.edges()

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c.to(torch.float32)
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def degree_drop_weights(g):
    g_ = g
    edge_ = g_.edges()
    in_deg = g_.in_degrees()
    deg_col = in_deg[edge_[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
    return weights


def pr_drop_weights(g, aggr: str = 'sink', k: int = 10):
    pv = compute_pr(g, k=k)
    edge_index = g.edges()
    pv_row = pv[edge_index[0]].to(torch.float32)
    pv_col = pv[edge_index[1]].to(torch.float32)
    s_row = torch.log(pv_row)
    s_col = torch.log(pv_col)
    if aggr == 'sink':
        s = s_col
    elif aggr == 'source':
        s = s_row
    elif aggr == 'mean':
        s = (s_col + s_row) * 0.5
    else:
        s = s_col
    weights = (s.max() - s) / (s.max() - s.mean())
    return weights


def evc_drop_weights(g):
    evc = eigenvector_centrality(g)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = g.edges()
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col
    return (s.max() - s) / (s.max() - s.mean())


def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_index = torch.vstack((edge_index[0], edge_index[1]))
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]


def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.
    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def drop_node(graph, drop_rate, return_mask_nodes=True):
    if drop_rate <= 0:
        return graph, None

    nodes = graph.nodes()
    N = graph.num_nodes()
    mask_rates = torch.FloatTensor(np.ones(N) * drop_rate)
    masks = torch.bernoulli(mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    remove_node = nodes[mask_idx]
    ng = dgl.remove_nodes(graph, remove_node)
    ng = ng.add_self_loop()

    nodes_list = nodes.tolist()
    mask_list = mask_idx.tolist()
    remove_list = list(set(nodes_list).difference(mask_list))

    if return_mask_nodes:
        return ng, remove_list
    else:
        return ng, None
