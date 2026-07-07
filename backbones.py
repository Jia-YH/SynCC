"""Alternative GNN backbones (SAGE / GIN / GCN / DotGAT).

These are kept so ``setup_module`` in :mod:`model` can build non-GAT
encoders/decoders. The default SynCC configuration uses GAT, so they are
inactive in the standard pipeline but preserved verbatim for completeness.
"""
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import SAGEConv
from dgl.nn.functional import edge_softmax
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader

from config import create_activation, create_norm, NormLayer


# ----------------------------------------------------------------------------
# GraphSAGE
# ----------------------------------------------------------------------------
class SAGE(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout,
                 activation, norm, aggr, encoding=False):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.active = create_activation(activation)
        last_activation = create_activation(activation) if encoding else None
        last_norm = norm(out_dim) if encoding else None

        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim, aggr))
        else:
            self.layers.append(SAGEConv(in_dim, num_hidden, aggr))
            for l in range(1, num_layers - 1):
                self.layers.append(SAGEConv(num_hidden, num_hidden, aggr))
            self.layers.append(SAGEConv(num_hidden, out_dim, aggr, norm=last_norm, activation=last_activation))

        self.head = nn.Identity()

    def forward(self, sub, x, return_hidden=False):
        h = x
        hidden_list = []
        for l, layer in enumerate(self.layers):
            h = layer(sub, h)
            if l != self.num_layers - 1:
                h = self.active(h)
                h = self.dropout(h)
                hidden_list.append(h)
        hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def inference(self, g, device, batch_size):
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g, torch.arange(g.num_nodes()).to(g.device), sampler,
            device=device, batch_size=batch_size, shuffle=False, drop_last=False,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != self.num_layers - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


# ----------------------------------------------------------------------------
# GCN
# ----------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout,
                 activation, residual, norm, encoding=False):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gcn_layers.append(GraphConv(
                in_dim, out_dim, residual=last_residual, norm=last_norm, activation=last_activation))
        else:
            self.gcn_layers.append(GraphConv(
                in_dim, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            for l in range(1, num_layers - 1):
                self.gcn_layers.append(GraphConv(
                    num_hidden, num_hidden, residual=residual, norm=norm, activation=create_activation(activation)))
            self.gcn_layers.append(GraphConv(
                num_hidden, out_dim, residual=last_residual, activation=last_activation, norm=last_norm))

        self.norms = None
        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.gcn_layers[l](g, h)
            if self.norms is not None and l != self.num_layers - 1:
                h = self.norms[l](h)
            hidden_list.append(h)
        if self.norms is not None and len(self.norms) == self.num_layers:
            h = self.norms[-1](h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, norm=None, activation=None, residual=True):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim

        self.fc = nn.Linear(in_dim, out_dim)

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_feats, self._out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.norm = norm
        if norm is not None:
            self.norm = norm(out_dim)
        self._activation = activation

        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            degs = graph.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_src.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat_src = feat_src * norm

            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
            rst = graph.dstdata['h']

            rst = self.fc(rst)

            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = rst * norm

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.norm is not None:
                rst = self.norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


# ----------------------------------------------------------------------------
# GIN
# ----------------------------------------------------------------------------
class GIN(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout,
                 activation, residual, norm, encoding=False, learn_eps=False, aggr="sum"):
        super(GIN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout

        last_activation = create_activation(activation) if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        if num_layers == 1:
            apply_func = MLP(2, in_dim, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, norm=norm, activation=activation)
            self.layers.append(GINConv(in_dim, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))
        else:
            self.layers.append(GINConv(
                in_dim, num_hidden,
                ApplyNodeFunc(MLP(2, in_dim, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                init_eps=0, learn_eps=learn_eps, residual=residual))
            for l in range(1, num_layers - 1):
                self.layers.append(GINConv(
                    num_hidden, num_hidden,
                    ApplyNodeFunc(MLP(2, num_hidden, num_hidden, num_hidden, activation=activation, norm=norm), activation=activation, norm=norm),
                    init_eps=0, learn_eps=learn_eps, residual=residual))
            apply_func = MLP(2, num_hidden, num_hidden, out_dim, activation=activation, norm=norm)
            if last_norm:
                apply_func = ApplyNodeFunc(apply_func, activation=activation, norm=norm)
            self.layers.append(GINConv(num_hidden, out_dim, apply_func, init_eps=0, learn_eps=learn_eps, residual=last_residual))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GINConv(nn.Module):
    def __init__(self, in_dim, out_dim, apply_func, aggregator_type="sum",
                 init_eps=0, learn_eps=False, residual=False):
        super().__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.apply_func = apply_func

        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))

        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, graph, feat):
        with graph.local_scope():
            aggregate_fn = fn.copy_src('h', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            return rst


class ApplyNodeFunc(nn.Module):
    def __init__(self, mlp, norm="batchnorm", activation="relu"):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        norm_func = create_norm(norm)
        if norm_func is None:
            self.norm = nn.Identity()
        else:
            self.norm = norm_func(self.mlp.output_dim)
        self.act = create_activation(activation)

    def forward(self, h):
        h = self.mlp(h)
        h = self.norm(h)
        h = self.act(h)
        return h


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)


# ----------------------------------------------------------------------------
# DotGAT
# ----------------------------------------------------------------------------
class DotGAT(nn.Module):
    def __init__(self, in_dim, num_hidden, out_dim, num_layers, nhead, nhead_out,
                 activation, feat_drop, attn_drop, residual, norm,
                 concat_out=False, encoding=False):
        super(DotGAT, self).__init__()
        self.out_dim = out_dim
        self.num_heads = nhead
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.concat_out = concat_out

        last_activation = create_activation(activation) if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(DotGatConv(
                in_dim, out_dim, nhead_out,
                feat_drop, attn_drop, last_residual, norm=last_norm, concat_out=concat_out))
        else:
            self.gat_layers.append(DotGatConv(
                in_dim, num_hidden, nhead,
                feat_drop, attn_drop, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            for l in range(1, num_layers - 1):
                self.gat_layers.append(DotGatConv(
                    num_hidden * nhead, num_hidden, nhead,
                    feat_drop, attn_drop, residual, create_activation(activation), norm=norm, concat_out=concat_out))
            self.gat_layers.append(DotGatConv(
                num_hidden * nhead, out_dim, nhead_out,
                feat_drop, attn_drop, last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.num_heads * self.out_dim, num_classes)


class DotGatConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 residual, activation=None, norm=None, concat_out=False,
                 allow_zero_in_degree=False):
        super(DotGatConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        self._concat_out = concat_out

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.activation = activation

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, self._out_feats * self._num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, self._out_feats * self._num_heads, bias=False)

        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)

        self.norm = norm
        if norm is not None:
            self.norm = norm(num_heads * out_feats)

    def forward(self, graph, feat, get_attention=False):
        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise ValueError('There are 0-in-degree nodes in the graph, '
                                 'output for those nodes will be invalid. '
                                 'Adding self-loop on the input graph by '
                                 'calling `g = dgl.add_self_loop(g)` will resolve '
                                 'the issue.')

        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            feat = self.feat_drop(feat)
            h_src = feat
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

        graph.srcdata.update({'ft': feat_src})
        graph.dstdata.update({'ft': feat_dst})

        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'] / self._out_feats ** 0.5)
        graph.edata["sa"] = self.attn_drop(graph.edata["sa"])
        graph.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))

        rst = graph.dstdata['agg_u']

        if self.res_fc is not None:
            batch_size = feat.shape[0]
            resval = self.res_fc(h_dst).view(batch_size, -1, self._out_feats)
            rst = rst + resval

        if self._concat_out:
            rst = rst.flatten(1)
        else:
            rst = torch.mean(rst, dim=1)

        if self.norm is not None:
            rst = self.norm(rst)

        if self.activation:
            rst = self.activation(rst)

        if get_attention:
            return rst, graph.edata['sa']
        else:
            return rst
