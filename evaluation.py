import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn import metrics
from munkres import Munkres


# ---------------------------------------------------------------------------
# Cluster-accuracy (Hungarian matching)
# ---------------------------------------------------------------------------
def cluster_acc(y_true, y_pred):
    if hasattr(y_true, 'cpu'):
        y_true = y_true.cpu().numpy()
    y_true = y_true.astype(int)
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)
    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('n_cluster is not valid')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    return acc, f1_macro, f1_micro


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for j in range(n_clusters):
        s = np.sum(C[:, j])
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(cluster_assignments, y_true, n_clusters):
    confusion = confusion_matrix(y_true, cluster_assignments, labels=None)
    cost_matrix = calculate_cost_matrix(confusion, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


# ---------------------------------------------------------------------------
# Imputation error
# ---------------------------------------------------------------------------
def cos_sim(x, y):
    sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return sim


def imputation_error(X_hat, X, drop_index):
    if drop_index is not None:
        i, j, ix = drop_index['i'], drop_index['j'], drop_index['ix']
        all_index = i[ix], j[ix]

        if isinstance(X_hat, torch.Tensor):
            x = X_hat[all_index].detach().cpu().numpy() if X_hat.is_cuda else X_hat[all_index].detach().numpy()
        else:
            x = X_hat[all_index]

        if isinstance(X, torch.Tensor):
            y = X[all_index].detach().cpu().numpy() if X.is_cuda else X[all_index].detach().numpy()
        else:
            y = X[all_index]

        squared_error = (x - y) ** 2
        absolute_error = np.abs(x - y)

        rmse = np.mean(np.sqrt(squared_error))
        median_l1_distance = np.median(absolute_error)
        cosine_similarity = cos_sim(x, y)
    else:
        rmse = 1
        median_l1_distance = 1
        cosine_similarity = 1
    return rmse, median_l1_distance, cosine_similarity


# ---------------------------------------------------------------------------
# Clustering evaluation
# ---------------------------------------------------------------------------
def clustering_for_transductive(model, graph, x, num_classes, cell_type, x_rec):
    model.eval()
    X = model.embed(graph, x)

    labels = cell_type
    labels = labels.cpu().detach().numpy()
    X = X.cpu().detach().numpy()

    pred = KMeans(n_clusters=num_classes, max_iter=100, n_init=10, init='k-means++',
                  algorithm='auto', random_state=0).fit_predict(X)
    y_pred = get_y_preds(pred, labels, num_classes)
    nmi = NMI(labels, y_pred)
    ari = adjusted_rand_score(labels, y_pred)
    acc = adjusted_rand_score(labels, y_pred)
    ca, reduced_ma_f1, reduced_mi_f1 = cluster_acc(cell_type, y_pred)
    print(f"--- Clustering NMI: {nmi:.4f}, Clustering ari: {ari:.4f}, Clustering ca: {ca:.4f} ")
    print("y_pred", len(y_pred))

    return nmi, acc, ari, ca
