from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
import scipy.io as sio
from torch_geometric.utils import to_scipy_sparse_matrix

name_list = ["ogbl-citation2", "ogbn-mag", "ogbn-products"]
root = "./dataset"

for name in name_list:
    if name in ["ogbn-mag", "ogbn-products"]:
        dataset = PygNodePropPredDataset(name = name, root = root)
    elif name in ["ogbl-citation2"]:
        dataset = PygLinkPropPredDataset(name = name, root = root)
    else:
        print("Unsupported dataset")
        exit()

    data = dataset[0]

    if name == "ogbn-mag":
        data = Data(
            x=data.x_dict['paper'],
            edge_index=data.edge_index_dict[('paper', 'cites', 'paper')],
            y=data.y_dict['paper'])

    MtxToScipy = to_scipy_sparse_matrix(data.edge_index)
    sio.mmwrite(f"code/FusedMM/dataset/{name}.mtx", MtxToScipy)