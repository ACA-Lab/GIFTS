import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from dgl.nn.pytorch import SumPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.profile import timeit
from dgl.utils import check_eq_shape, expand_as_pair

# GCN convolution along the graph structure, we follow codes provided by OGB team in pcqm4m conv.py (url: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb_lsc/PCQM4M), since it's the only offcial implementation of GCN for OGB in DGL. We make some necessary changes to make it comparable in node missions.
class DGL_GCNConv(nn.Module):
    def __init__(self, in_feats,
        out_feats):
        """
        emb_dim (int): node embedding dimensionality
        """

        super(DGL_GCNConv, self).__init__()

        self.linear = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))
        self.register_parameters()

    def register_parameters(self):
        init.xavier_uniform_(self.linear)
        init.zeros_(self.bias)
        

    def forward(self, g, x):
        with g.local_scope():
            with timeit(log=False) as comb:
                x = torch.matmul(x, self.linear)
            
            with timeit(log=False) as aggr:
                degs = (g.out_degrees().float() + 1).to(x.device)
                norm = torch.pow(degs, -0.5)
                shape = norm.shape + (1,) * (x.dim() - 1)
                norm = torch.reshape(norm, shape)
                x = x * norm

                g.srcdata["x"] = x
                g.update_all(fn.copy_u("x", "m"), fn.sum("m", "new_x"))
                out = g.ndata["new_x"]

            out = out + self.bias

            print(f"Aggregation Time: {aggr.duration}")
            # print(f"Combination Time: {comb.duration}")

            return out
        
class DGL_SAGEConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type = "mean",
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(DGL_SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)

        if aggregator_type != "gcn":
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        elif bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if self._aggre_type != "gcn":
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Message Passing
            if self._aggre_type == "mean":
                with timeit(log=False) as comb:
                    graph.srcdata["h"] = (
                        self.fc_neigh(feat_src)
                    )
                with timeit(log=False) as aggr:
                    graph.update_all(msg_fn, fn.mean("m", "neigh"))
                h_neigh = graph.dstdata["neigh"]
            else:
                raise KeyError(
                    "Aggregator type {} not recognized.".format(
                        self._aggre_type
                    )
                )

            print(f"Aggregation Time: {aggr.duration}")
            # print(f"Combination Time: {comb.duration}")

            rst = self.fc_self(h_self) + h_neigh

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
