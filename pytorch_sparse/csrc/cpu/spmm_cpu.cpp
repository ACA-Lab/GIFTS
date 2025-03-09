#include "spmm_cpu.h"

#include <ATen/Parallel.h>
#include <omp.h>
#include <cstdio>

#include "reducer.h"
#include "utils.h"

#define FloatZero 1e-5
#define ABS(x) ((x)>0?(x):-(x))
/* Cacheblock size 64s, int64 takes 8 Bytes, so there're 8 int64 per block */
#define CacheBlockInt64 8
#define CacheBlockInt16 32
#define PrefetchLookAhead 16
/* High Degree Threshold */
#define HighDegThres 500

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_cpu", [&] {
    scalar_t *value_data = nullptr;
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            b = i / M, m = i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              c = col_data[e];
              if (HAS_VALUE)
                val = value_data[e];
              for (auto k = 0; k < K; k++) {
                if (HAS_VALUE)
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], val * mat_data[offset + c * K + k], &args[k],
                      e);
                else
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], mat_data[offset + c * K + k], &args[k], e);
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_reorder_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, torch::Tensor Reorder_seq) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto seq_data = Reorder_seq.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_cpu", [&] {
    scalar_t *value_data = nullptr;
    auto mat_data = mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            auto _i = seq_data[i];
            b = _i / M, m = _i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            if (i%(CacheBlockInt64) == 0){
              __builtin_prefetch(&rowptr_data[seq_data[i+PrefetchLookAhead]], 0, 2);
            }

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              c = col_data[e];
              if (HAS_VALUE)
                val = value_data[e];
              for (auto k = 0; k < K; k++) {
                if (HAS_VALUE)
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], val * mat_data[offset + c * K + k], &args[k],
                      e);
                else
                  Reducer<scalar_t, REDUCE>::update(
                      &vals[k], mat_data[offset + c * K + k], &args[k], e);
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}

/* OpenMP version only support SUM reduce */
std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_openmp_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  using scalar_t = float;
  scalar_t *value_data = nullptr;
  auto mat_data = mat.data_ptr<scalar_t>();
  auto out_data = out.data_ptr<scalar_t>();

  const bool HAS_VALUE = optional_value.has_value();
  
  if (HAS_VALUE) {
    value_data = optional_value.value().data_ptr<scalar_t>();
  }
  printf("Openmp dynamic\n");
#pragma omp parallel for schedule(dynamic)
  for (auto i = 0; i < B * M; i++) {
    for (auto k = 0; k < K; k++)
      out_data[i * K + k] = 0;

    for (auto e = rowptr_data[i]; e < rowptr_data[i+1]; e++) {
      for (auto k = 0; k < K; k++) {
        if (HAS_VALUE)
          out_data[i * K + k] += value_data[e] * mat_data[col_data[e] * K + k];
        else
          out_data[i * K + k] += mat_data[col_data[e] * K + k];
      }
    }
    if (reduce == "mean"){
      for (auto k = 0; k < K; k++) {
        out_data[i * K + k] /= rowptr_data[i+1] - rowptr_data[i];
      }
    }
  }
  return std::make_tuple(out, arg_out);
}

torch::Tensor spmm_value_bw_cpu(torch::Tensor row, torch::Tensor rowptr,
                                torch::Tensor col, torch::Tensor mat,
                                torch::Tensor grad, std::string reduce) {
  CHECK_CPU(row);
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(mat);
  CHECK_CPU(grad);

  mat = mat.contiguous();
  grad = grad.contiguous();

  auto M = grad.size(-2);
  auto N = mat.size(-2);
  auto E = row.numel();
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  auto out = torch::zeros({row.numel()}, grad.options());

  auto row_data = row.data_ptr<int64_t>();
  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_value_bw_cpu", [&] {
    auto mat_data = mat.data_ptr<scalar_t>();
    auto grad_data = grad.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    scalar_t val;
    int64_t row, col;
    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      for (int b = 0; b < B; b++) {
        for (int e = 0; e < E; e++) {
          row = row_data[e], col = col_data[e], val = (scalar_t)0;
          for (int k = 0; k < K; k++) {
            val += mat_data[b * N * K + col * K + k] *
                   grad_data[b * M * K + row * K + k];
          }
          if (REDUCE == MEAN) {
            int row_start = rowptr_data[row], row_end = rowptr_data[row + 1];
            val /= (scalar_t)std::max(row_end - row_start, 1);
          }
          out_data[e] += val;
        }
      }
    });
  });

  return out;
}

torch::Tensor
bal_workload_cpu(torch::Tensor rowptr, int64_t thread_num, int64_t block_size){
    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto node_num = rowptr.numel() - 1;
    auto deg_list = torch::empty({node_num}, rowptr.options());
    auto deg_list_data = deg_list.data_ptr<int64_t>();
    int64_t grain_size = at::internal::GRAIN_SIZE;
    /* Calculate Degrees */
    at::parallel_for(0, node_num, grain_size, [&](int64_t begin, int64_t end){
        for (auto i = begin; i < end; i++){
            deg_list_data[i] = rowptr_data[i+1] - rowptr_data[i];
        }
    });
    auto nodes_per_thread = node_num / thread_num;
    auto blocks_per_thread = nodes_per_thread / block_size;
    auto block_num = blocks_per_thread * thread_num;
    auto deg_block_list = torch::zeros({block_num}, rowptr.options());
    auto deg_block_list_data = deg_block_list.data_ptr<int64_t>();
    /* Calculate Degrees for each block */
    at::parallel_for(0, thread_num, 0, [&](int64_t begin, int64_t end){
        for (auto i = begin; i < end; i++){
            for (auto j = 0; j < blocks_per_thread; j++){
                for (auto k = 0; k < block_size; k++)
                    deg_block_list_data[i*blocks_per_thread + j] += deg_list_data[i*nodes_per_thread + j*block_size + k]; 
            }
        }
    });
    auto cnt_per_thread = torch::zeros({thread_num}, rowptr.options());
    auto cnt_per_thread_data = cnt_per_thread.data_ptr<int64_t>();
    /* Init with nodes not in blocks within each thread */
    at::parallel_for(0, thread_num, 0, [&](int64_t begin, int64_t end){
        for (auto i = begin; i < end; i++){
            for (auto j = i*nodes_per_thread + block_size*blocks_per_thread; j < (i+1)*nodes_per_thread; j++){
                cnt_per_thread_data[i] += deg_list_data[j];
            }
        }
    });
    std::tuple<torch::Tensor,torch::Tensor> sorted_value = torch::sort(deg_block_list, -1, 1);
    auto value_sorted = std::get<0>(sorted_value);
    auto idx_sorted = std::get<1>(sorted_value);
    auto bal_block_seq = torch::empty({block_num}, rowptr.options());
    auto bal_block_seq_data = bal_block_seq.data_ptr<int64_t>();
    /* Scheduling Algorithm, here we use OpenMP since there's a reduction */
// #pragma omp parallel for reduction(+:cnt_per_thread_data)
    for (auto i = 0; i < blocks_per_thread; i++){
        torch::Tensor sorted_cnt = torch::argsort(cnt_per_thread);
        for (auto j = 0; j < thread_num; j++){
            auto temp = sorted_cnt[j].item().toLong();
            bal_block_seq_data[temp*blocks_per_thread + i] = idx_sorted[i*thread_num + j].item().toLong();
            cnt_per_thread_data[temp] += value_sorted[i*thread_num + j].item().toLong();
        }
    }
    auto bal_res = torch::arange({node_num}, rowptr.options());
    auto bal_res_data = bal_res.data_ptr<int64_t>();
    /* Project block order to node order */
    at::parallel_for(0, block_num, grain_size/256, [&](int64_t begin, int64_t end){
        for (auto i = 0; i < block_num; i++){
            auto thread_idx = i / blocks_per_thread;
            auto block_idx = i % blocks_per_thread;
            auto bal_th_idx = bal_block_seq_data[i] / blocks_per_thread;
            auto bal_bl_idx = bal_block_seq_data[i] % blocks_per_thread;
            for (auto j = 0; j < block_size; j++){
                auto _addr = thread_idx*nodes_per_thread + block_idx*block_size;
                auto _val = bal_th_idx* nodes_per_thread + bal_bl_idx*block_size;
                bal_res_data[_addr+j] = _val+j;
            }
        }        
    });
    return bal_res;
}

torch::Tensor
get_deg_list_cpu(torch::Tensor rowptr){
    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto node_num = rowptr.numel() - 1;
    auto deg_list = torch::empty({node_num}, rowptr.options());
    auto deg_list_data = deg_list.data_ptr<int64_t>();
    int64_t grain_size = at::internal::GRAIN_SIZE;
    /* Calculate Degrees */
    at::parallel_for(0, node_num, grain_size, [&](int64_t begin, int64_t end){
        for (auto i = begin; i < end; i++){
            deg_list_data[i] = rowptr_data[i+1] - rowptr_data[i];
        }
    });
    return deg_list;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_cpu(torch::Tensor rowptr, torch::Tensor mat){
  mat = mat.contiguous();
  auto sizes = mat.sizes().vec();
  auto num_node = int32_t(sizes[0]);
  auto node_size = int16_t(sizes[1]);
  auto mat_data = mat.data_ptr<float>();
  auto comp_val = torch::zeros(sizes, torch::dtype(torch::kBFloat16));
  auto comp_val_data = comp_val.data_ptr<at::BFloat16>();
  auto comp_idx = torch::zeros(sizes, torch::dtype(torch::kInt16));
  auto comp_idx_data = comp_idx.data_ptr<int16_t>();
  int64_t grain_size = at::internal::GRAIN_SIZE;
  /* Compressed degrees */
  auto HDN_deg = torch::zeros({num_node}, torch::dtype(torch::kInt16));
  auto HDN_deg_data = HDN_deg.data_ptr<int16_t>();
  at::parallel_for(0, num_node, grain_size / node_size, [&](int64_t begin, int64_t end){
      for (auto i = begin; i < end; i++){
        for (auto j = 0; j < node_size; j++){
          if (ABS(mat_data[i*node_size + j]) > FloatZero){
            comp_val_data[i*node_size + HDN_deg_data[i]] = mat_data[i*node_size + j];
            comp_idx_data[i*node_size + HDN_deg_data[i]] = j;
            HDN_deg_data[i] += 1;
          } 
        }
      }
  });
  // comp_val = comp_val.to(torch::kBFloat16);
  // comp_idx = comp_idx.to(torch::kInt16);
  return std::make_tuple(HDN_deg, comp_idx, comp_val);
}

/* Compress sparse feature only for High Degree Nodes (HDN) */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_HDN_cpu(torch::Tensor colptr, torch::Tensor mat, int64_t high_thres){
  mat = mat.contiguous();
  auto colptr_data = colptr.data_ptr<int64_t>();
  int64_t grain_size = at::internal::GRAIN_SIZE;
  auto sizes = mat.sizes().vec();
  auto num_node = int32_t(sizes[0]);
  auto node_size = int16_t(sizes[1]);
  // printf("num_node: %d, node_size: %d\n", num_node, node_size);
  auto HDN_deg = torch::zeros({num_node}, torch::dtype(torch::kInt16));
  auto HDN_deg_data = HDN_deg.data_ptr<int16_t>();
  auto mat_data = mat.data_ptr<float>();
  auto comp_val = torch::zeros(sizes, torch::dtype(torch::kBFloat16));
  auto comp_val_data = comp_val.data_ptr<at::BFloat16>();
  auto comp_idx = torch::zeros(sizes, torch::dtype(torch::kInt16));
  auto comp_idx_data = comp_idx.data_ptr<int16_t>();
  // for (int i = 0; i < 10; i++)
  //   printf("mat_data[%d]: %f\n", i, mat_data[i]);
  at::parallel_for(0, num_node, grain_size / node_size, [&](int64_t begin, int64_t end){
      for (auto i = begin; i < end; i++){
        if (colptr_data[i+1] - colptr_data[i] > high_thres){
          for (auto j = 0; j < node_size; j++){
            if (ABS(mat_data[i*node_size + j]) > FloatZero){
              comp_val_data[i*node_size + HDN_deg_data[i]] = mat_data[i*node_size + j];
              comp_idx_data[i*node_size + HDN_deg_data[i]] += j;
              HDN_deg_data[i] += 1;
            }
          }
        }
        else{
          for (auto j = 0; j < node_size; j++){
            comp_val_data[i*node_size + j] = mat_data[i*node_size + j];
          }
        }
      }
  });
  // comp_val = comp_val.to(torch::kBFloat16);
  // comp_idx = comp_idx.to(torch::kInt16);
  return std::make_tuple(HDN_deg, comp_idx, comp_val);
}

/* Compress sparse feature only for High Degree Nodes (HDN) without bit-level compression */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compress_SpFeature_HDN_value_only_cpu(torch::Tensor colptr, torch::Tensor mat, int64_t high_thres){
  mat = mat.contiguous();
  auto colptr_data = colptr.data_ptr<int64_t>();
  int64_t grain_size = at::internal::GRAIN_SIZE;
  auto sizes = mat.sizes().vec();
  auto num_node = int64_t(sizes[0]);
  auto node_size = int64_t(sizes[1]);
  // printf("num_node: %d, node_size: %d\n", num_node, node_size);
  auto HDN_deg = torch::zeros({num_node}, torch::dtype(torch::kInt64));
  auto HDN_deg_data = HDN_deg.data_ptr<int64_t>();
  auto mat_data = mat.data_ptr<float>();
  auto comp_val = torch::zeros(sizes, torch::dtype(torch::kFloat32));
  auto comp_val_data = comp_val.data_ptr<float>();
  auto comp_idx = torch::zeros(sizes, torch::dtype(torch::kInt64));
  auto comp_idx_data = comp_idx.data_ptr<int64_t>();
  // for (int i = 0; i < 10; i++)
  //   printf("mat_data[%d]: %f\n", i, mat_data[i]);
  at::parallel_for(0, num_node, grain_size / node_size, [&](int64_t begin, int64_t end){
      for (auto i = begin; i < end; i++){
        if (colptr_data[i+1] - colptr_data[i] > high_thres){
          for (auto j = 0; j < node_size; j++){
            if (ABS(mat_data[i*node_size + j]) > FloatZero){
              comp_val_data[i*node_size + HDN_deg_data[i]] = mat_data[i*node_size + j];
              comp_idx_data[i*node_size + HDN_deg_data[i]] += j;
              HDN_deg_data[i] += 1;
            }
          }
        }
        else{
          for (auto j = 0; j < node_size; j++){
            comp_val_data[i*node_size + j] = mat_data[i*node_size + j];
          }
        }
      }
  });
  // comp_val = comp_val.to(torch::kBFloat16);
  // comp_idx = comp_idx.to(torch::kInt16);
  return std::make_tuple(HDN_deg, comp_idx, comp_val);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  auto HDN_deg = std::get<0>(CompressTuple);
  auto HDN_deg_data = HDN_deg.data_ptr<int16_t>();
  auto comp_idx = std::get<1>(CompressTuple);
  auto comp_idx_data = comp_idx.data_ptr<int16_t>();
  auto comp_val = std::get<2>(CompressTuple);
  auto comp_val_data = comp_val.data_ptr<at::BFloat16>();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_cpu", [&] {
    using scalar_t = float;
    scalar_t *value_data = nullptr;
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            b = i / M, m = i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              if (e%(CacheBlockInt16) == 0){
                __builtin_prefetch(&HDN_deg_data[col_data[e+CacheBlockInt16]], 0, 2);
              }
              c = col_data[e];
              if (HAS_VALUE)
                  val = value_data[e];
              if (HDN_deg_data[c] > 0){
                auto HDN_deg_num = HDN_deg_data[c];
                for (auto k = 0; k < HDN_deg_num; k++) {
                  auto val_out_idx = comp_idx_data[offset + c * K + k];
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
              else{  
                for (auto k = 0; k < K; k++) {
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}
std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_compress_value_only_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);

  auto HDN_deg = std::get<0>(CompressTuple);
  auto HDN_deg_data = HDN_deg.data_ptr<int64_t>();
  auto comp_idx = std::get<1>(CompressTuple);
  auto comp_idx_data = comp_idx.data_ptr<int64_t>();
  auto comp_val = std::get<2>(CompressTuple);
  auto comp_val_data = comp_val.data_ptr<float>();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_cpu", [&] {
    using scalar_t = float;
    scalar_t *value_data = nullptr;
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            b = i / M, m = i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              if (e%(CacheBlockInt16) == 0){
                __builtin_prefetch(&HDN_deg_data[col_data[e+CacheBlockInt16]], 0, 2);
              }
              c = col_data[e];
              if (HAS_VALUE)
                  val = value_data[e];
              if (HDN_deg_data[c] > 0){
                auto HDN_deg_num = HDN_deg_data[c];
                for (auto k = 0; k < HDN_deg_num; k++) {
                  auto val_out_idx = comp_idx_data[offset + c * K + k];
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
              else{  
                for (auto k = 0; k < K; k++) {
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}

std::tuple<torch::Tensor, torch::optional<torch::Tensor>>
spmm_reorder_compress_cpu(torch::Tensor rowptr, torch::Tensor col,
         torch::optional<torch::Tensor> optional_value, torch::Tensor mat,
         std::string reduce, torch::Tensor Reorder_seq, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CompressTuple) {
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value())
    CHECK_CPU(optional_value.value());
  CHECK_CPU(mat);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  if (optional_value.has_value()) {
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().size(0) == col.size(0));
  }
  CHECK_INPUT(mat.dim() >= 2);

  mat = mat.contiguous();

  auto sizes = mat.sizes().vec();
  sizes[mat.dim() - 2] = rowptr.numel() - 1;
  auto out = torch::empty(sizes, mat.options());

  torch::optional<torch::Tensor> arg_out = torch::nullopt;
  int64_t *arg_out_data = nullptr;
  if (reduce2REDUCE.at(reduce) == MIN || reduce2REDUCE.at(reduce) == MAX) {
    arg_out = torch::full_like(out, col.numel(), rowptr.options());
    arg_out_data = arg_out.value().data_ptr<int64_t>();
  }

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto seq_data = Reorder_seq.data_ptr<int64_t>();

  auto M = rowptr.numel() - 1;
  auto N = mat.size(-2);
  auto K = mat.size(-1);
  auto B = mat.numel() / (N * K);
  auto HDN_deg = std::get<0>(CompressTuple);
  auto HDN_deg_data = HDN_deg.data_ptr<int16_t>();
  auto comp_idx = std::get<1>(CompressTuple);
  auto comp_idx_data = comp_idx.data_ptr<int16_t>();
  auto comp_val = std::get<2>(CompressTuple);
  auto comp_val_data = comp_val.data_ptr<at::BFloat16>();

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, mat.scalar_type(), "spmm_cpu", [&] {
    using scalar_t = float;
    scalar_t *value_data = nullptr;
    auto out_data = out.data_ptr<scalar_t>();

    AT_DISPATCH_REDUCTION_TYPES(reduce, [&] {
      AT_DISPATCH_HAS_VALUE(optional_value, [&] {
        if (HAS_VALUE) {
          value_data = optional_value.value().data_ptr<scalar_t>();
        }

        int64_t grain_size = at::internal::GRAIN_SIZE /
                             (K * std::max(col.numel() / M, (int64_t)1));
        at::parallel_for(0, B * M, grain_size, [&](int64_t begin, int64_t end) {
          scalar_t val;
          std::vector<scalar_t> vals(K);
          int64_t row_start, row_end, b, m, c;
          std::vector<int64_t> args(K);

          for (auto i = begin; i < end; i++) {
            auto _i = seq_data[i];
            b = _i / M, m = _i % M;

            row_start = rowptr_data[m], row_end = rowptr_data[m + 1];

            if (i%(CacheBlockInt64) == 0){
              __builtin_prefetch(&rowptr_data[seq_data[i+PrefetchLookAhead]], 0, 2);
            }

            for (auto k = 0; k < K; k++)
              vals[k] = Reducer<scalar_t, REDUCE>::init();

            auto offset = b * N * K;
            for (auto e = row_start; e < row_end; e++) {
              if (e%(CacheBlockInt16) == 0){
                __builtin_prefetch(&HDN_deg_data[col_data[e+CacheBlockInt16]], 0, 2);
              }
              c = col_data[e];
              if (HAS_VALUE)
                  val = value_data[e];
              if (HDN_deg_data[c] > 0){
                auto HDN_deg_num = HDN_deg_data[c];
                for (auto k = 0; k < HDN_deg_num; k++) {
                  auto val_out_idx = comp_idx_data[offset + c * K + k];
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[val_out_idx], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
              else{  
                for (auto k = 0; k < K; k++) {
                  if (HAS_VALUE)
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], val * comp_val_data[offset + c * K + k], &args[k],
                        e);
                  else
                    Reducer<scalar_t, REDUCE>::update(
                        &vals[k], comp_val_data[offset + c * K + k], &args[k], e);
                }
              }
            }
            offset = b * M * K + m * K;
            for (auto k = 0; k < K; k++)
              Reducer<scalar_t, REDUCE>::write(out_data + offset + k, vals[k],
                                               arg_out_data + offset + k,
                                               args[k], row_end - row_start);
          }
        });
      });
    });
  });

  return std::make_tuple(out, arg_out);
}