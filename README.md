This is the official implementation of the following paper:

> **GIFTS: Efficient GCN Inference Framework on PyTorch-CPU via Exploring the Sparsity**
>
> Ruiyang Chen, Xing Li, Xiaoyao Liang, and Zhuoran Song

## Code Base Structure

```
$CODE_DIR
├── code            # source code for GNN inference
├── dataset         # dataset (need downloading at the first time)
├── Excel           # processed results (using proc_data.py)
├── MISC            # stored checkpoint for GNN models
├── model           # GNN conv layers (including DGL, PyG and GIFTS!)
├── pytorch_sparse  # source code for GIFTS (need compling at the first time)
└── Result          # experiment results
```

## Environment Setup

Before you begin, ensure that you have Anaconda or Miniconda installed on your system. Please run

```bash
conda env create --file environment.yml
```

to install python packages and run

```bash
cd pytorch_sparse
bash compile.sh
```

to compile GIFTS's kernels (it usually takes 5 to 10 minutes).

---

Please run

```bash
conda activate GIFTS
```

to activate GIFTS's conda environment.

## Reproduce Our Results

We provide an easy-to-use script for our main results, that is the speedup of variants of GIFTS against TorchSparse and DistGNN.

Please run

```bash
python run.py
```

The evaluation will begin soon after the datasets are ready, note that those large-scale datasets may need several hours to be downloaded at the first time.

The whole evaluation may take 1 to 2 hours (except time for data downloading), and you can see results in the `Result` directory. It is recommended to use `htop` to monitor the process.

After all results are produced, please run

```bash
python proc_data.py
```

to get the final results in Excel format (.xlsx) for better viewing. The data in the Excel represents the time for each method (`TorchSparse` and `DistGNN` are compared SOTAs, `Combined` is the final version of `GIFTS`). Note that execution time will be affected by several factors (e.g. hardware architecture, frequency, SMT, etc.), so it is not a const value.

## Integrating GIFTS into Your Code

We provide an easy-to-use GCN convolution layer that allows you to seamlessly replace your existing GCN layers with the following code:

```python
from model import GIFTS_GCNConv as GCNConv, GIFTS_SAGEConv as SAGEConv
# Your Code Here
```

Additionally, you can refer to `code/Ours/ogbn-products/test_inference_gnn.py` for GCN and SAGE implementations:

```python
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, CompressFlag=False, ReorderSeq=None, ValueFlag=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False,  
                    PrintFlag=PrintFlag, ReorderSeq=ReorderSeq, CompressFlag=CompressFlag, SparseFlag=False, ValueFlag=ValueFlag))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, normalize=False, 
                        PrintFlag=PrintFlag, ReorderSeq=ReorderSeq, CompressFlag=CompressFlag, ValueFlag=ValueFlag))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False, 
                    PrintFlag=PrintFlag, ReorderSeq=ReorderSeq, CompressFlag=CompressFlag, ValueFlag=ValueFlag))
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
```

We also offer an interface for balancing thread workloads. You can use the following code to obtain the scheduled order:

```python
from torch_sparse import bal_workload
rowptr, _, _ = data.adj_t.csr()
ReorderSeq = bal_workload(rowptr, args.cpu_threads, args.block_size)
model = GCN(..., ReorderSeq=ReorderSeq, ...).to(device)
```

## FAQs

1. **DGL’s ogbl-citation2 Results.** If you create a conda environment using `environment.yml`, you may encounter issues running DistGNN (DGL) on ogbl-citation2. This is because PyG 2.3.0 lacks the `to_dgl()` function. In our experiments, we used PyG 2.5.1, but it caused version conflicts when creating a one-click conda environment. Therefore, we selected PyG 2.3.0 for compatibility with other libraries. To resolve this, you can manually install PyG via `pip install torch_geometric` or try initializing conda with our experimental environment (`environment_real.yml`), though our tests showed failures due to some dependencies being installed via pip.
2. **Reproducing FusedMM.** Since FusedMM is not compatible with mainstream frameworks like PyG and DGL, we reproduce it using the author’s provided code (`code/FusedMM`). First, we convert PyG datasets into a format supported by FusedMM using our automated script (`generate_FusedMM_data.py`). Next, you need to compile the FusedMM kernel, following the steps in `code/FusedMM/README.md`. Finally, we provide a test script (`run_FusedMM.py`) that automates result generation.

## Citation

To cite our work, please use

```bibtex
@inproceedings{chen2025gifts,
  abbr      = {IPDPS},
  author    = {Ruiyang Chen and Xing Li and Xiaoyao Liang and Zhuoran Song},
  title     = {GIFTS: Efficient GCN Inference Framework on PyTorch-CPU via Exploring the Sparsity},
  booktitle = {Proceedings of the IEEE International Parallel \& Distributed Processing Symposium (IPDPS)},
  year      = {2025},
}
```
