"""SynCC model: GAT masked graph auto-encoder with contrastive + ZINB heads.

Reorganized from the original ``graphmae/models/edcoder.py``.

IMPORTANT: ``PreModel.__init__`` is preserved verbatim (identical submodule
names, shapes and construction) so that checkpoints trained with the original
code load with ``strict=True`` and inference is numerically unchanged.
Only clearly-dead methods (the standalone ``clustering`` fine-tuner, the legacy
``encoding_mask_noise``, ``sage_embed``, ``reconstruct_adj`` etc.) were removed.
"""
from typing import Optional
from functools import partial

import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn as dglnn

from config import create_norm
from backbones import SAGE, GIN, GCN, DotGAT
from graph_ops import (
    drop_edge, drop_node,
    degree_drop_weights, pr_drop_weights, evc_drop_weights,
    feature_drop_weights, compute_pr, eigenvector_centrality,
    drop_edge_weighted, drop_feature_weighted,
)
from layers import MeanAct, DispAct, ZINBLoss
from losses import sce_loss


########################################
# GAT (dgl.nn.GATConv based)
########################################
class GAT(nn.Module):
    def __init__(self, in_dim: int, num_hidden: int, out_dim: int, num_layers: int,
                 nhead: int, nhead_out: int, concat_out: bool, activation, feat_drop: float,
                 attn_drop: float, negative_slope: float, residual: bool, norm: str,
                 encoding: bool):
        super(GAT, self).__init__()

        if isinstance(activation, str):
            if activation.lower() == 'relu':
                activation = F.relu
            elif activation.lower() == 'prelu':
                activation = nn.PReLU()
            elif activation.lower() == 'gelu':
                activation = F.gelu
            else:
                raise NotImplementedError(f"Activation {activation} is not supported.")

        self.activation = activation
        self.layers = nn.ModuleList()
        self.concat_out = concat_out

        self.layers.append(dglnn.GATConv(in_dim, num_hidden, nhead,
                                         feat_drop=feat_drop, attn_drop=attn_drop,
                                         negative_slope=negative_slope,
                                         residual=residual, activation=activation))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GATConv(num_hidden * nhead, num_hidden, nhead,
                                             feat_drop=feat_drop, attn_drop=attn_drop,
                                             negative_slope=negative_slope,
                                             residual=residual, activation=activation))
        self.layers.append(dglnn.GATConv(num_hidden * nhead, out_dim, nhead_out,
                                         feat_drop=feat_drop, attn_drop=attn_drop,
                                         negative_slope=negative_slope,
                                         residual=residual, activation=None))
        if norm:
            self.norm_layer = create_norm(norm)(out_dim)
        else:
            self.norm_layer = None

    def forward(self, g, feat, return_hidden=False):
        hidden_states = []
        h = feat
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = h.flatten(1)
            else:
                if self.concat_out:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)
            if self.norm_layer:
                h = self.norm_layer(h)
            hidden_states.append(h)
        if return_hidden:
            return h, hidden_states
        return h


########################################
# setup_module: build encoder/decoder backbone
########################################
def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, aggr, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        if enc_dec == "encoding":
            # encoder: input 2000, hidden 512//4, output 128, 4 heads
            mod = GAT(
                in_dim=2000, num_hidden=512 // 4, out_dim=128,
                num_layers=num_layers, nhead=4, nhead_out=4, concat_out=True,
                activation=activation, feat_drop=dropout, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual,
                norm=create_norm(norm), encoding=True)
        else:
            # decoder: input 512, hidden 1024, output 2000, 2 heads
            mod = GAT(
                in_dim=512, num_hidden=1024, out_dim=2000,
                num_layers=2, nhead=2, nhead_out=nhead_out, concat_out=True,
                activation=activation, feat_drop=dropout, attn_drop=attn_drop,
                negative_slope=negative_slope, residual=residual,
                norm=create_norm(norm), encoding=False)
    elif m_type == "sage":
        mod = SAGE(in_dim=in_dim, num_hidden=num_hidden, out_dim=out_dim,
                   num_layers=num_layers, activation=activation, dropout=dropout,
                   norm=create_norm(norm), aggr=aggr, encoding=(enc_dec == "encoding"))
    elif m_type == "dotgat":
        mod = DotGAT(in_dim=in_dim, num_hidden=num_hidden, out_dim=out_dim,
                     num_layers=num_layers, nhead=nhead, nhead_out=nhead_out,
                     concat_out=concat_out, activation=activation, feat_drop=dropout,
                     attn_drop=attn_drop, residual=residual, norm=create_norm(norm),
                     encoding=(enc_dec == "encoding"))
    elif m_type == "gin":
        mod = GIN(in_dim=in_dim, num_hidden=num_hidden, out_dim=out_dim,
                  num_layers=num_layers, dropout=dropout, activation=activation,
                  residual=residual, norm=norm, encoding=(enc_dec == "encoding"))
    elif m_type == "gcn":
        mod = GCN(in_dim=in_dim, num_hidden=num_hidden, out_dim=out_dim,
                  num_layers=num_layers, dropout=dropout, activation=activation,
                  residual=residual, norm=create_norm(norm), encoding=(enc_dec == "encoding"))
    elif m_type == "mlp":
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden), nn.PReLU(), nn.Dropout(0.2), nn.Linear(num_hidden, out_dim))
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


########################################
# PreModel
########################################
class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_projector_hidden: int,
            num_projector: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            temperature: float = 0.4,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            loss_weight: float = 0.5,
            mu: float = 0.5,
            nu: float = 0.5,
            num_classes: int = 8,
            max_epoch: int = 200,
            batch_size: int = 1024,
            rec_weight: float = 1.0,
            con_weight: float = 0.2,
            zinb_weight: float = 0.8,
            augmentation: str = "drop_node",
            drop_node_rate: float = 0.0,
            drop_edge_rate: float = 0.0,
            drop_feature_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            aggr: str = None,
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._temperature = temperature

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._augmentation = augmentation
        self._drop_node_rate = drop_node_rate
        self._drop_edge_rate = drop_edge_rate
        self._drop_feature_rate = drop_feature_rate
        self._output_hidden_size = num_hidden
        self._num_projector = num_projector
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._loss_weight = loss_weight
        self._mu = mu
        self._nu = nu
        self.rec_weight = rec_weight
        self.con_weight = con_weight
        self.zinb_weight = zinb_weight
        self.max_epoch = max_epoch
        self.batch_size = batch_size

        # fixed encoder/decoder dims
        self.encoder_out_dim = 512
        self.decoder_in_dim = 512
        self.decoder_hidden = 1024

        hidden = 512
        nnm = 512

        if not concat_hidden:
            self.ZINB_Decoder = nn.Sequential(
                nn.Linear(hidden, nnm), nn.BatchNorm1d(nnm), nn.PReLU(),
                nn.Dropout(0.2), nn.Linear(nnm, 256), nn.BatchNorm1d(256), nn.PReLU())
        else:
            self.ZINB_Decoder = nn.Sequential(
                nn.Linear(hidden * num_layers, nnm), nn.BatchNorm1d(nnm), nn.PReLU(),
                nn.Dropout(0.2), nn.Linear(nnm, 256), nn.BatchNorm1d(256), nn.PReLU())

        self.mean_layer = nn.Sequential(nn.Linear(256, in_dim), MeanAct())
        self.disp_layer = nn.Sequential(nn.Linear(256, in_dim), DispAct())
        self.pi_layer = nn.Sequential(nn.Linear(256, in_dim), nn.Sigmoid())

        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = 512
            enc_nhead = 4
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        self.encoder = setup_module(
            m_type=encoder_type, enc_dec="encoding", in_dim=2000, num_hidden=512,
            out_dim=self.encoder_out_dim, num_layers=3, nhead=4, nhead_out=enc_nhead,
            concat_out=True, activation=activation, dropout=feat_drop, attn_drop=attn_drop,
            negative_slope=negative_slope, residual=residual, norm=norm, aggr=aggr)

        self.decoder = setup_module(
            m_type=decoder_type, enc_dec="decoding", in_dim=self.decoder_in_dim,
            num_hidden=self.decoder_hidden, out_dim=2048, num_layers=2, nhead=2,
            nhead_out=nhead_out, concat_out=True, activation=activation, dropout=feat_drop,
            attn_drop=attn_drop, negative_slope=negative_slope, residual=residual,
            norm=norm, aggr=aggr)

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.std_expander = nn.Sequential(nn.Linear(512, 512), nn.PReLU())

        self.encoder_to_decoder = nn.Linear(self.encoder_out_dim, 512, bias=False)

        self.projector_fc1 = nn.Sequential(
            nn.Linear(512, 256, bias=True), nn.PReLU(), nn.Linear(256, 128, bias=True))
        self.projector_fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=True), nn.PReLU(), nn.Linear(256, 128, bias=True))
        self.projector_fc3 = nn.Sequential(
            nn.Linear(512, 256, bias=True), nn.PReLU(), nn.Linear(256, 128, bias=True))

        self.shared_proj_base = nn.Sequential(nn.Linear(512, 256, bias=True), nn.PReLU())

        self.view1_head = nn.Linear(256, 128, bias=True)
        self.view2_head = nn.Linear(256, 128, bias=True)

        self.intra_projector1 = nn.Sequential(
            nn.Linear(512, 256, bias=True), nn.PReLU(), nn.Linear(256, 128, bias=True))
        self.intra_projector2 = nn.Sequential(
            nn.Linear(512, 256, bias=True), nn.PReLU(), nn.Linear(256, 128, bias=True))

        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)
        self.sorted_indices = None

    # ---------------- contrastive helpers ----------------
    def nt_xent_loss(self, z1, z2):
        """NT-Xent loss over two views, each of shape [N, d]."""
        batch_size = z1.size(0)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.t())
        logits = similarity_matrix / self._temperature

        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        logits = logits.masked_fill(mask, -1e9)

        positive_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z1.device),
            torch.arange(0, batch_size, device=z1.device)])

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        loss = -log_prob[torch.arange(2 * batch_size, device=z1.device), positive_indices]
        return loss

    def contrastive_loss(self, emb, enc_edge_intra):
        base1 = self.shared_proj_base(emb)
        base2 = self.shared_proj_base(enc_edge_intra)
        h1 = F.normalize(self.view1_head(base1), p=2, dim=1)
        h2 = F.normalize(self.view1_head(base2), p=2, dim=1)
        return self.nt_xent_loss(h1, h2).mean()

    def contrastive_loss_ture(self, emb1, emb2):
        h1 = self.intra_projector1(emb1)
        h2 = self.intra_projector2(emb2)
        h1 = F.normalize(h1, p=2, dim=1)
        h2 = F.normalize(h2, p=2, dim=1)
        return self.nt_xent_loss(h1, h2).mean()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    # ---------------- masking / scoring ----------------
    def encoding_mask_noise_update(self, g, x, sorted_indices, epoch, mask_rate=0.3):
        max_epoch = self.max_epoch
        num_nodes = g.num_nodes()
        num_mask_nodes = int(mask_rate * num_nodes)
        B = mask_rate
        A = 1 - mask_rate

        easy_nodes = sorted_indices[-int(A * num_nodes):]
        hard_nodes = sorted_indices[:int(B * num_nodes)]

        def get_weights(epoch, max_epoch):
            easy = 0.2 + 0.8 * (1 - epoch / max_epoch)
            hard = 0.1 + 0.9 * (epoch / max_epoch)
            return easy, hard

        easy_weight, hard_weight = get_weights(epoch, max_epoch)

        mask_probs = np.ones(num_nodes) * 0.1
        mask_probs[easy_nodes] = easy_weight * 0.6
        mask_probs[hard_nodes] = hard_weight * 0.8
        mask_probs = np.clip(mask_probs, 0.05, 0.95)

        selected_indices = np.random.choice(
            num_nodes, size=num_mask_nodes, p=mask_probs / mask_probs.sum(), replace=False)

        mask_nodes = torch.tensor(selected_indices, device=x.device)
        all_indices = torch.arange(num_nodes, device=x.device)
        keep_nodes = all_indices[~torch.isin(all_indices, mask_nodes)]
        out_x = x.clone()
        noise = torch.randn_like(x[mask_nodes]) * 0.1
        out_x[mask_nodes] = x[mask_nodes] * 0.5 + noise
        out_x[mask_nodes] += self.enc_mask_token

        return g.clone(), out_x, (mask_nodes, keep_nodes)

    def get_score(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()

        degrees = g.in_degrees().float().cpu().numpy()
        pr = self._pagerank_dgl(g).cpu().numpy()
        G_nx = self._build_networkx_graph(g)
        bc_dict = nx.betweenness_centrality(G_nx, k=min(100, num_nodes))
        bc = np.array([bc_dict.get(i, 0) for i in range(num_nodes)], dtype=np.float32)

        def normalize(arr):
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        degree_norm = normalize(degrees)
        pr_norm = normalize(pr)
        bc_norm = normalize(bc)

        scores = 0.3 * degree_norm + 0.5 * pr_norm + 0.2 * bc_norm
        sorted_indices = np.argsort(scores)
        return sorted_indices

    def _pagerank_dgl(self, g, alpha=0.85, max_iter=100, tol=1e-6):
        num_nodes = g.num_nodes()
        device = g.device

        pr = torch.ones(num_nodes, device=device) / num_nodes
        out_degree = g.out_degrees().float().clamp(min=1)

        u, v = g.edges()
        edge_weights = 1.0 / out_degree[u]
        g.edata['weight'] = edge_weights

        for _ in range(max_iter):
            g.ndata['pr'] = pr
            g.update_all(fn.u_mul_e('pr', 'weight', 'm'), fn.sum('m', 'pr_sum'))
            pr_new = alpha * g.ndata['pr_sum'] + (1 - alpha) / num_nodes

            if torch.norm(pr_new - pr) < tol:
                break
            pr = pr_new
        return pr

    def _build_networkx_graph(self, g):
        u, v = g.edges()
        edge_list = zip(u.cpu().numpy(), v.cpu().numpy())
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G.add_nodes_from(range(g.num_nodes()))
        return G

    # ---------------- forward / training ----------------
    def forward(self, g, x, celltype, epoch):
        num_nodes = x.size(0)
        bs = self.batch_size
        losses = []
        agg_components = {k: 0.0 for k in [
            "loss_rec", "loss_contrastive_true", "loss_contrastive",
            "loss_zinb", "loss_adj", "loss_std"]}
        recs = []
        mean = []
        disp = []
        pi = []
        for start in range(0, num_nodes, bs):
            end = min(start + bs, num_nodes)
            batch_idx = torch.arange(start, end, device=x.device)

            sub_g = dgl.node_subgraph(g, batch_idx)
            sub_g = dgl.add_self_loop(sub_g)

            x_batch = x[batch_idx]
            celltype_batch = celltype[batch_idx]

            loss_b, comp_b, x_rec_b, mean_b, disp_b, pi_b = self.mask_attr_prediction(
                sub_g, x_batch, celltype_batch, epoch)

            losses.append(loss_b)
            recs.append(x_rec_b)
            mean.append(mean_b)
            disp.append(disp_b)
            pi.append(pi_b)
            for k, v in comp_b.items():
                agg_components[k] += v

        loss = torch.stack(losses).mean()
        n_batches = len(losses)
        for k in agg_components:
            agg_components[k] /= n_batches

        x_rec = torch.cat(recs, dim=0)
        loss_item = {"loss": loss.item()}
        return loss, loss_item, agg_components, x_rec, mean, disp, pi

    def mask_attr_prediction(self, g, x, celltype, epoch):
        if epoch == 0:
            self.sorted_indices = self.get_score(g, x, self._mask_rate)
            pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise_update(
                g, x, self.sorted_indices, epoch, self._mask_rate)
        else:
            pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise_update(
                g, x, self.sorted_indices, epoch, self._mask_rate)

        if self._augmentation == 'drop_node':
            aug_g, aug_x, (drop_idx, keep) = self.encoding_mask_noise_update(
                g, x, self.sorted_indices, epoch, self._mask_rate)
        elif self._augmentation in ['degree', 'pr', 'evc']:
            num_nodes = g.num_nodes()
            device = g.device
            g_cpu = g.to('cpu')
            x_cpu = g_cpu.ndata['feat']
            if self._augmentation == 'degree':
                g_ = g_cpu
                node_c = g_.in_degrees()
                drop_weights = degree_drop_weights(g_)
                feature_weights = feature_drop_weights(x_cpu, node_c)
            elif self._augmentation == 'pr':
                node_c = compute_pr(g_cpu)
                drop_weights = pr_drop_weights(g_cpu, aggr='source', k=200)
                feature_weights = feature_drop_weights(x_cpu, node_c)
            elif self._augmentation == 'evc':
                node_c = eigenvector_centrality(g_cpu)
                drop_weights = evc_drop_weights(g_cpu)
                feature_weights = feature_drop_weights(x_cpu, node_c)
            else:
                drop_weights = None
                feature_weights = None
            edge_ = drop_edge_weighted(g_cpu.edges(), drop_weights, self._drop_edge_rate, threshold=0.7)
            x_ = drop_feature_weighted(x_cpu, feature_weights, self._drop_feature_rate)
            aug_g = dgl.graph((edge_[0], edge_[1]), num_nodes=num_nodes)
            aug_g = aug_g.add_self_loop()
            aug_g.ndata['feat'] = x_
            aug_g = aug_g.to(device)
            aug_x = x_.to(device)
        else:
            raise NotImplementedError

        enc_rep, all_hidden = self.encoder(pre_use_g, use_x, return_hidden=True)
        enc_edge, all_edge_hidden = self.encoder(aug_g, aug_x, return_hidden=True)
        aug_g_intra, _ = drop_edge(aug_g, self._drop_edge_rate, return_edges=True)
        enc_edge_intra, _ = self.encoder(aug_g_intra, aug_x, return_hidden=True)
        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        rep = self.encoder_to_decoder(enc_rep)
        rep_copy = rep
        if self._decoder_type not in ("mlp", "linear"):
            rep[mask_nodes] = 0

        rep_z = self.ZINB_Decoder(rep_copy)
        mean = self.mean_layer(rep_z)
        disp = self.disp_layer(rep_z)
        pi = self.pi_layer(rep_z)
        zinb_criterion = ZINBLoss().to(g.device)
        zinb_loss = zinb_criterion(x, mean, disp, pi)

        recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes].to(g.device)
        x_rec = recon[mask_nodes].to(g.device)

        loss_rec = self.criterion(x_rec, x_init)
        loss = (
            self.rec_weight * loss_rec +
            self._loss_weight * self.contrastive_loss_ture(enc_edge, enc_rep) +
            self.con_weight * self.contrastive_loss(enc_edge, enc_edge_intra) +
            self.zinb_weight * zinb_loss +
            self._mu * self.reconstruct_adj_mse(g, enc_rep) +
            self._nu * self.std_loss(enc_rep)
        )
        loss_components = {
            "loss_rec": loss_rec.item(),
            "loss_contrastive_true": self.contrastive_loss_ture(enc_edge, enc_rep).item(),
            "loss_contrastive": self.contrastive_loss(enc_edge, enc_edge_intra).item(),
            "loss_zinb": zinb_loss.item(),
            "loss_adj": self.reconstruct_adj_mse(g, enc_rep).item(),
            "loss_std": self.std_loss(enc_rep).item()
        }

        return loss, loss_components, recon, mean, disp, pi

    # ---------------- inference / regularizers ----------------
    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def reconstruct_adj_mse(self, g, emb):
        adj = g.adj().to_dense().to(emb.device)
        res_adj = torch.sigmoid(emb @ emb.t())
        relative_distance = (adj * res_adj).sum() / (res_adj * (1 - adj)).sum()
        cri = nn.MSELoss()
        res_loss = cri(adj, res_adj) + F.binary_cross_entropy_with_logits(adj, res_adj)
        loss = res_loss + relative_distance
        return loss

    def std_loss(self, z):
        z = self.std_expander(z)
        z = F.normalize(z, dim=1)
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z))
        return std_loss


def build_model(args, num_features, num_classes):
    model = PreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_projector_hidden=args.num_projector_hidden,
        num_projector=args.num_projector,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        nhead_out=args.num_out_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=args.negative_slope,
        residual=args.residual,
        encoder_type=args.encoder,
        decoder_type=args.decoder,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        norm=args.norm,
        loss_fn=args.loss_fn,
        loss_weight=args.loss_weight,
        mu=args.mu,
        nu=args.nu,
        num_classes=num_classes,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
        rec_weight=args.rec_weight,
        con_weight=args.con_weight,
        zinb_weight=args.zinb_weight,
        augmentation=args.augmentation,
        drop_node_rate=args.drop_node_rate,
        drop_edge_rate=args.drop_edge_rate,
        drop_feature_rate=args.drop_feature_rate,
        replace_rate=args.replace_rate,
        alpha_l=args.alpha_l,
        concat_hidden=args.concat_hidden,
        aggr=args.aggr,
    )
    return model
