#include <cuda_tuntime.h>
#define BLOCK_SIZE 8
__global__ void matrix_transpose_kernel(const float* input, float* output,
                                        int cols) {
  __shared__ float scratch[BLOCK_SIZE][BLOCK_SIZE + 1];
  const int in_x = threadIdx.x + BLOCK_SIZE * blockIdx.x;
  const int in_y = threadIdx.y + BLOCK_SIZE * blockIdx.y;
  const int in_cols = cols;
  const int in_rows = rows;

  const int input_idx = in_x + in_cols * in_y;
  if (input_idx < rows * cols) {
    scratch[threadIdx.y][threadIdx.x] = input[input_idx];
  }
  __syncthreads();
  const int out_y = threadIdx.y + BLOCK_SIZE * blockIdx.x;  // NOTE 很容易理解！
  const int out_x = threadIdx.x + BLOCK_SIZE * blockIdx.y;
  const int out_cols = rows;
  const int out_rows = cols;
  if (out_x < out_cols && out_y < out_rows) {
    output[out_x + out_cols * out_y] = scratch[threadIdx.x][threadIdx.y];
  }
}
