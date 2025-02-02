#include "gpt.h"
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_reduce.cuh>
#include <torch/extension.h>

__global__ void gelu_kernel(float *out, const float *in, int num_rows,
                            int num_cols) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // For reference the torch formula is
  // out = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (in + 0.044715
  // * torch.pow(x, 3.0))))
}

template <int NUM_THREADS>
__global__ void layernorm_kernel(float *out, const float *in,
                                 const float *weights, const float *bias,
                                 float eps, int num_rows, int num_cols) {
  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  using TempStorage = typename BlockReduce::TempStorage;
  __shared__ TempStorage temp_storage;
  __shared__ float mean_shared;
  __shared__ float var_shared;

  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Step 1: Compute the average and variance
  // using the fact that Var[x] = E[x^2] - E[x]^2
  // - do a local sum by iterating on chunks of size blockDim.x
  // - use BlockReduce to perform a block-wide sum, example:
  //   ```
  //   float total = BlockReduce(temp_storage).Sum(sum);
  //   __syncthreads();
  //   ```
  //   more information on
  //   https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html#_CPPv4N3cub11BlockReduce3SumE1T

  // Step 3: compute layernorm and write output, see
  // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
}

__global__ void linear_kernel(float *out, const float *in, const float *weights,
                              const float *bias, int m, int k, int n) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  // Perform a matrix multiplication.
  // All tensors are stored in row-major order
  // but /!\ weights is transposed
}

// NOTE: all those kernel launcher are suggestions, feel free to modify them
void gelu(torch::Tensor out, torch::Tensor in) {
  dim3 block_size = 1024;
  dim3 grid_size{((unsigned int)in.size(1) + block_size.x - 1) / block_size.x,
                 (unsigned int)in.size(0)};
  auto stream = c10::cuda::getCurrentCUDAStream();

  gelu_kernel<<<grid_size, block_size, 0, stream>>>(
      out.data_ptr<float>(), in.data_ptr<float>(), in.size(0), in.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layernorm(torch::Tensor out, torch::Tensor in, torch::Tensor weights,
               torch::Tensor bias, float eps) {
  // NOTE: We only launch one block per line
  dim3 constexpr block_size = 1024;
  dim3 grid_size{1, (unsigned int)in.size(0)};
  auto stream = c10::cuda::getCurrentCUDAStream();

  layernorm_kernel<block_size.x><<<grid_size, block_size, 0, stream>>>(
      out.data_ptr<float>(), in.data_ptr<float>(), weights.data_ptr<float>(),
      bias.data_ptr<float>(), eps, in.size(0), in.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void linear(torch::Tensor out, torch::Tensor in, torch::Tensor weights,
            torch::Tensor bias) {
  int m = in.size(0);
  int k = in.size(1);
  int n = out.size(1);

  dim3 block_size{32, 32};
  dim3 grid_size{(n + block_size.x - 1) / block_size.x,
                 (m + block_size.y - 1) / block_size.y};
  auto stream = c10::cuda::getCurrentCUDAStream();
  size_t shared_mem_size = 0;

  linear_kernel<<<grid_size, block_size, shared_mem_size, stream>>>(
      out.data_ptr<float>(), in.data_ptr<float>(), weights.data_ptr<float>(),
      bias.data_ptr<float>(), m, k, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
