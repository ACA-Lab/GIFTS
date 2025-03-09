# from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data

# name = "ogbn-mag"
name = "ogbl-citation2"
root = "./dataset"

# dataset = PygNodePropPredDataset(name = name, root = root)
dataset = PygLinkPropPredDataset(name = name, root = root)

data = dataset[0]

if name == "ogbn-mag":
    data = Data(
        x=data.x_dict['paper'],
        edge_index=data.edge_index_dict[('paper', 'cites', 'paper')],
        y=data.y_dict['paper'])

import scipy.io as sio
from torch_geometric.utils import to_scipy_sparse_matrix

MtxToScipy = to_scipy_sparse_matrix(data.edge_index)
sio.mmwrite(f"code/FusedMM/dataset/{name}.mtx", MtxToScipy)