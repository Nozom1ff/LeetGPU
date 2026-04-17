#include <cuda_runtime.h>
#include <stdio.h>
__global__ void vectorAddOne(float *d_out, const float *d_in, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    d_out[idx] = d_in[idx] + 1.0f;
  }
}

int main() {
  const int N = 1024 * 100;
  size_t bytes = N * sizeof(float);
  // NOTE 1.分配主机内存
  float *h_in, *h_out;
  h_in = (float *)malloc(bytes);
  h_out = (float *)malloc(bytes);

  for (int i = 0; i < N; i++) {
    h_in[i] = (float)i;
  }

  // NOTE 2. device_memory
  float *d_in, *d_out;
  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  // H2D
  cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostTodevice);
  int blockSize = 256;
  int gridSize = (N + blockSize - 1);
  vectorrAddOne<<<gridSize, blockSize>>>(d_out, d_in, N);
  // NOTE 3.
  cudaGetLastError();
  // NOTE 4 sync
  cudaDeviceSynchronize();

  // D2H
  cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
  // NOTE5 释放
  cudaFree(d_in);
  cudaFree(d_out);

  // ==============================
  // 11. 释放 CPU 内存
  // ==============================
  free(h_in);
  free(h_out);

  printf("CUDA 程序执行完毕！\n");
  return 0;
}
