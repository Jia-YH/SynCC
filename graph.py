"""kNN cell-graph construction (verbatim from the original ``knn_graph``)."""
import torch
import torch.nn.functional as F
import dgl


def knn_graph(embeddings, k, gcn_norm=False, sym=True, remove_self_loop=True):
    """Build a kNN graph from node embeddings.

    Args:
        embeddings: node embeddings ``[N, D]`` (may live on GPU)
        k: number of neighbours
    """
    device = embeddings.device
    N = embeddings.shape[0]

    embeddings = F.normalize(embeddings, dim=1, p=2)
    similarity = torch.mm(embeddings, embeddings.t())  # [N, N]

    _, topk_indices = torch.topk(similarity, k=k + 1, dim=1)  # include self

    src = torch.repeat_interleave(torch.arange(N, device=device), repeats=k + 1)
    dst = topk_indices.flatten()

    if remove_self_loop:
        mask = src != dst
        src, dst = src[mask], dst[mask]

    g = dgl.graph((src, dst), num_nodes=N, device=device)

    if sym:
        g_cpu = g.to('cpu')
        g_cpu = dgl.to_bidirected(g_cpu, copy_ndata=True)
        g = g_cpu.to(device)

    return g
