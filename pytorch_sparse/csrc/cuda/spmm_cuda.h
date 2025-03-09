#pragma once

#include "../extensions.h"

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_cuda(torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_reorder_cuda(torch::Tensor reorder_seq, torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_reorder_cuda(torch::Tensor reorder_seq, torch::Tensor rowptr, torch::Tensor col,
          torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
          std::string reduce);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_cuda(torch::Tensor mat){
    auto mat_data = mat.data_ptr<float>();
    auto sizes = mat.sizes().vec();
    auto comp_mat = torch::zeros(sizes, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
    auto comp_mat_data = comp_mat.data_ptr<at::BFloat16>();
    cudaMemcpy(comp_mat_data, mat_data, mat.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
    auto empty_tensor_1 = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto empty_tensor_2 = torch::empty({0}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    return std::make_tuple(empty_tensor_1, empty_tensor_2, comp_mat);
};

torch::Tensor spmm_value_bw_cuda(torch::Tensor row, torch::Tensor rowptr,
                                 torch::Tensor col, torch::Tensor mat,
                                 torch::Tensor grad, std::string reduce);
