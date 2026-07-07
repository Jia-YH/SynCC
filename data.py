import os

import numpy as np
import scipy.sparse
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder

from graph import knn_graph


def drop_data(adata, rate, datatype='real'):
    """Randomly zero-out a fraction ``rate`` of nonzero entries (imputation eval).

    With ``rate == 0`` no entry is dropped, so ``obsm['train'] == obsm['test']``.
    """
    X = adata.X

    if scipy.sparse.issparse(X):
        X = np.array(X.todense())

    if datatype == 'real':
        X_train = np.copy(X)
        i, j = np.nonzero(X)

        ix = np.random.choice(range(len(i)), int(np.floor(rate * len(i))), replace=False)
        X_train[i[ix], j[ix]] = 0.0

        drop_index = {'i': i, 'j': j, 'ix': ix}
        adata.uns['drop_index'] = drop_index
        adata.obsm["train"] = X_train
        adata.obsm["test"] = X

        # for training
        adata.raw.X[i[ix], j[ix]] = 0.0

    elif datatype == 'simul':
        adata.obsm["train"] = X

    return adata


class Trainer:
    """Loads an ``.h5ad`` dataset, preprocesses it and builds the kNN graph."""

    def __init__(self, args):
        self.args = args
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"

        self.data_path = f'dataset/{self.args.name}.h5ad'
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        self._init_dataset()

        self.args.n_nodes = self.adata.obsm["train"].shape[0]
        self.args.n_feat = self.adata.obsm["train"].shape[1]
        self.args.feat = self.adata.obsm["train"]

    def _init_dataset(self):
        self.adata = sc.read(self.data_path)

        if self.adata.obs['celltype'].dtype != int:
            self.label_encoding()

        self.preprocess(HVG=self.args.HVG, size_factors=self.args.sf,
                        logtrans_input=self.args.log, normalize_input=self.args.normal)
        self.adata = drop_data(self.adata, rate=self.args.drop_rate)

    def label_encoding(self):
        label_encoder = LabelEncoder()
        celltype = self.adata.obs['celltype']
        celltype = label_encoder.fit_transform(celltype)
        self.adata.obs['celltype'] = celltype

    def preprocess(self, HVG=2000, size_factors=True, logtrans_input=True, normalize_input=False):
        sc.pp.filter_cells(self.adata, min_counts=1)
        sc.pp.filter_genes(self.adata, min_counts=1)

        if isinstance(self.adata.X, np.ndarray):
            variance = np.var(self.adata.X, axis=0)
        else:
            variance = np.array(self.adata.X.todense().var(axis=0))[0]

        hvg_gene_idx = np.argsort(variance)[-int(HVG):]
        self.adata = self.adata[:, hvg_gene_idx]
        self.adata.raw = self.adata.copy()

        if size_factors:
            sc.pp.normalize_per_cell(self.adata)
            self.adata.obs['size_factors'] = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
        else:
            self.adata.obs['size_factors'] = 1.0

        if logtrans_input:
            sc.pp.log1p(self.adata)

        if normalize_input:
            sc.pp.scale(self.adata)

    def data(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_raw = None
        drop_index = None

        if self.args.drop_rate != 0.0:
            X_raw = self.adata.obsm["test"]
            drop_index = self.adata.uns['drop_index']

        cell_data = torch.Tensor(self.adata.obsm["train"]).to(device)
        g = knn_graph(cell_data, self.args.k)

        features_train = self.args.feat
        num_features = self.args.n_feat = self.adata.obsm["train"].shape[1]

        cell_type = self.adata.obs['celltype'].values
        num_cell_type = np.unique(cell_type).shape[0]

        return g, num_features, features_train, cell_type, num_cell_type, X_raw, drop_index
