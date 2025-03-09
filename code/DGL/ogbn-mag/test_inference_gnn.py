import torch
import torch.nn.functional as F

import os
from tqdm.auto import tqdm
import argparse

import time
import random

import dgl
from torch_geometric.profile import timeit
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from model import DGL_GCNConv as GCNConv, DGL_SAGEConv as SAGEConv

### importing OGB-Node
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels))
        self.convs.append(
            GCNConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, nfeat):
        for conv in self.convs[:-1]:
            nfeat = conv(graph, nfeat)
            nfeat = F.relu(nfeat)
            nfeat = F.dropout(nfeat, p=self.dropout, training=self.training)
        nfeat = self.convs[-1](graph, nfeat)
        return torch.log_softmax(nfeat, dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, graph, nfeat):
        for conv in self.convs[:-1]:
            nfeat = conv(graph, nfeat)
            nfeat = F.relu(nfeat)
            nfeat = F.dropout(nfeat, p=self.dropout, training=self.training)
        nfeat = self.convs[-1](graph, nfeat)
        return torch.log_softmax(nfeat, dim=-1)

@torch.no_grad()
def test(model, graph, nfeat, labels, split_idx, evaluator):
    model.eval()

    out = model(graph, nfeat)
    y_pred = out.argmax(dim=-1, keepdim=True)
    labels = labels.unsqueeze(1)

    train_acc = evaluator.eval({
        'y_true': labels[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': labels[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': labels[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on mag with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of GNN message passing layers (default: 2)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 256)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to inference (default: 1)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '',
                        help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '',
                        help='directory to save test submission file')
    parser.add_argument('--dataset_route', type=str, default = './dataset',
                        help='dataset route (default: ./dataset)')
    parser.add_argument('--use_cpu_only', action='store_true',
                        help='only use cpu to do inference/training')
    parser.add_argument('--dataset_name', type=str, default = 'ogbn-mag',
                        help='dataset name (default: ogbn-mag')
    parser.add_argument('--cpu_threads', type=int, default=16,
                        help='num of cpu cores in inference (default: 16)')
    args = parser.parse_args()

    print(args)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    if args.use_cpu_only:
        device = torch.device("cpu")
        torch.set_num_threads(args.cpu_threads)
    elif torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    data = DglNodePropPredDataset(name = args.dataset_name, root = args.dataset_route)
    evaluator = Evaluator(name = args.dataset_name)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"]['paper'],
        splitted_idx["valid"]['paper'],
        splitted_idx["test"]['paper'],
    )
    graph, labels = data[0]

    nfeat = graph.nodes["paper"].data["feat"].to(device)
    labels = labels["paper"].to(device).squeeze()

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()
    graph = graph[('paper', 'cites', 'paper')]
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    if args.gnn == 'gcn':
        model = GCN(in_feats, args.emb_dim,
                    n_classes, args.num_layers,
                    args.drop_ratio).to(device)
    elif args.gnn == 'sage':
        model = SAGE(in_feats, args.emb_dim,
                    n_classes, args.num_layers,
                    args.drop_ratio).to(device)
    else:
        raise ValueError('Invalid GNN type')

    try:
        num_params = sum(p.numel() for p in model.parameters())
        print(f'#Params: {num_params}')
    except:
        num_params = -1
        print(f'Warning: can not init num_params')

    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f'Checkpoint file not found at {checkpoint_path}')
    
    ## reading in checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Predicting on test data...')
    for i in range(args.epochs):
        test(model, graph, nfeat, labels, splitted_idx, evaluator)
if __name__ == "__main__":
    main()
