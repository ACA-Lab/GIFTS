#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce);
         
std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_reorder_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, torch::Tensor Reorder_seq);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_openmp_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce);

torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
                                torch::Tensor col, torch::Tensor mat,
                                torch::Tensor grad, std::string reduce);

torch::Tensor
bal_workload_cpu(torch::Tensor rowptr, int64_t thread_num, int64_t block_size);

torch::Tensor
get_deg_list_cpu(torch::Tensor rowptr);

/* Out Params: HDN_deg, comp_idx, comp_val */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_cpu(torch::Tensor rowptr, torch::Tensor mat);

/* Compress sparse feature only for High Degree Nodes (HDN) */
/* Out Params: HDN_deg, comp_idx, comp_val */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_HDN_cpu(torch::Tensor rowptr, torch::Tensor mat, int64_t high_thres);

/* Compress sparse feature only for High Degree Nodes (HDN) without bit-level compression*/
/* Out Params: HDN_deg, comp_idx, comp_val */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_HDN_value_only_cpu(torch::Tensor rowptr, torch::Tensor mat, int64_t high_thres);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_value_only_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_reorder_compress_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, torch::Tensor Reorder_seq, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple);