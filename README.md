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