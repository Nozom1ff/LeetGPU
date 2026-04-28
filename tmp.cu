#include <cuda_runtime.h>

template <int BLOCK_SIZE>
__global__ void reduce_kernel(const float *input, float *output, int N) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  sdata[tid] = i < N ? input[i] : 0.0f;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0]; // 只用 thread0 写
  }
}

extern "C" void solve(const float *d_input, float *d_output, int N) {
  const int BLOCK_SIZE = 256;
  int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *temp;
  cudaMalloc(&temp, sizeof(float) * grid_size);

  reduce_kernel<256><<<grid_size, BLOCK_SIZE>>>(d_input, temp, N);
  int remaining = grid_size;
  while (remaining > 1) {
    int new_grid = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<256><<<new_grid, BLOCK_SIZE>>>(temp, temp, remaining);
    remaining = new_grid;
  }
  cudaMemcpy(d_output, temp, sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(temp);
}

#define scale 8
#define BLOCK_SIZE 256

__global__ void reducuction(const float *input, float *output, int N) {
  __shared__ float smem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int block_offset = blockDim.x * blockIdx.x * scale;
  float sum = 0.0f;
  for (int step = 0; step < scale; ++step) {
    int idx = block_offset + tid + blockDim.x * step;
    sum += (idx < N) ? input[idx] : 0.0f;
  }
  smem[tid] = sum;
  __syncthreads();
  if (BLOCK_SIZE >= 512 && tid < 256) {
    smem[tid] += smem[tid + 256];
  }
  __syncthreads();
  if (BLOCK_SIZE >= 256 && tid < 128) {
    smem[tid] += smem[tid + 128];
  }
  __syncthreads();
  if (BLOCK_SIZE > 128 && tid < 64) {
    smem[tid] += smem[tid + 64];
  }
  __syncthreads();
  if (tid < 32) {
    float val = smem[tid] + smem[tid + 32];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
  }
  if (tid == 0) {
    atomicAdd(output, val);
  }
}
