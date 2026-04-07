#include <cfloat>
#include <cuda_runtime.h>
__global__ void softmax_kernel(const float *input, float *output, int N) {
  int tid = threadIdx.x;

  float v_max = -FLT_MAX;
  for (int i = tid; i < N; i += blockDim.x) {
    v_max = max(input[i], v_max);
  }

  for (int offset = 16; offset >= 1; offset >>= 1) {
    v_max = max(v_max, __shfl_xor_sync(0xffffffff, v_max, offset));
  } // 不需要共享内存，通过异或操作直接同步通信了 v_max

  float v_sum = 0.0f;
  for (int i = tid; i < N; i += blockDim.x) {
    v_sum += __expf(input[i] - v_max);
  }

  for (int offset = 16; offset >= 1; offset >>= 1) {
    v_sum += __shfl_xor_sync(0xffffffff, v_sum, offset);
  }
  for (int i = tid; i < N; i += blockDim.x) {
    output[i] = __expf(input[i] - v_max) / v_sum;
  }
}
// 之所以 blockDim.x = 32 ，就是为了直接warp 内通信
extern "C" void solve(const float *input, float *output, int N) {

  int threadsPerBlock = 32;

  int blocksPerGrid = 1;

  softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);

  cudaDeviceSynchronize();
}
// 下面是 flashsoftmax 
