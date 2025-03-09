import torch

def bal_workload(rowptr, thread_num=32, block_size=16):
    return torch.ops.torch_sparse.bal_workload(rowptr, thread_num, block_size)

def get_deg_list(rowptr):
    return torch.ops.torch_sparse.get_deg_list(rowptr)

def compress_SpFeature(colptr, mat):
    return torch.ops.torch_sparse.compress_SpFeature(colptr, mat)

def compress_SpFeature_HDN(colptr, mat, high_thres=100):
    return torch.ops.torch_sparse.compress_SpFeature_HDN(colptr, mat, high_thres)

def compress_SpFeature_HDN_value_only(colptr, mat, high_thres=100):
    return torch.ops.torch_sparse.compress_SpFeature_HDN_value(colptr, mat, high_thres)