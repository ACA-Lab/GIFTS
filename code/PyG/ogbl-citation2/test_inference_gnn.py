import torch
import torch.nn.functional as F

import os
from tqdm.auto import tqdm
import argparse

import time
import random

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from model import GCNConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.profile import timeit
from torch.utils.data import DataLoader

### importing OGB-Link
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

PrintFlag=True

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False, 
                    PrintFlag=PrintFlag))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False,
                        PrintFlag=PrintFlag))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False,
                    PrintFlag=PrintFlag))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            SAGEConv(in_channels, hidden_channels, normalize=False, 
                     PrintFlag=PrintFlag))
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, normalize=False,
                         PrintFlag=PrintFlag))
        self.convs.append(
            SAGEConv(hidden_channels, out_channels, normalize=False,
                     PrintFlag=PrintFlag))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size):
    predictor.eval()

    h = model(data.x, data.adj_t)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(batch_size), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 2).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(batch_size*2), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 2)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on products with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of GNN message passing layers (default: 3)')
    parser.add_argument('--emb_dim', type=int, default=256,
                        help='dimensionality of hidden units in GNNs (default: 600)')
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
    parser.add_argument('--dataset_name', type=str, default = 'ogbn-products',
                        help='dataset name (default: ogbn-products')
    parser.add_argument('--cpu_threads', type=int, default=16,
                        help='num of cpu cores in inference (default: 16)')
    args = parser.parse_args()

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

    dataset = PygLinkPropPredDataset(name = args.dataset_name, root = args.dataset_route, transform = T.ToSparseTensor())

    data = dataset[0]

    split_edge = dataset.get_edge_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(name = args.dataset_name)

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    if args.gnn == 'gcn':
        model = GCN(data.num_features, args.emb_dim,
                    args.emb_dim, args.num_layers,
                    args.drop_ratio).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
    elif args.gnn == 'sage':
        model = SAGE(data.num_features, args.emb_dim,
                    args.emb_dim, args.num_layers,
                    args.drop_ratio).to(device)
    else:
        raise ValueError('Invalid GNN type')
    
    predictor = LinkPredictor(args.emb_dim, args.emb_dim, 1,
                              args.num_layers, args.drop_ratio).to(device)

    data = data.to(device)

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
    predictor.load_state_dict(checkpoint['predictor_state_dict'])

    print('Predicting on test data...')
    for i in range(args.epochs):
        test(model, predictor, data, split_edge, evaluator, args.batch_size)

if __name__ == "__main__":
    main()
