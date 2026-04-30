#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// 矩阵尺寸，保证计算量 & 访存量足够大
constexpr int N = 2048;
constexpr int BLOCK = 32;

// ==============================================
// 版本1：低效实现 —— 重复全局访存、计算强度低
// 特征：memory bound，Roofline 贴带宽线
// ==============================================
__global__ void naiveKernel(const float *A, const float *B, float *C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * N)
    return;

  // 多次重复读取全局内存，访存放大
  float aval = A[idx];
  float bval = B[idx];
  float res = 0.0f;

  // 少量计算 + 大量访存，算术强度极低
  for (int i = 0; i < 8; ++i) {
    res += aval * bval;
    res -= A[idx] + B[idx]; // 重复读全局
    res *= 0.99f;
  }
  C[idx] = res;
}

// ==============================================
// 版本2：优化实现 —— 复用寄存器、合并访存、提升计算密度
// 特征：compute bound，Roofline 贴近算力峰值
// ==============================================
__global__ void optKernel(const float *__restrict__ A,
                          const float *__restrict__ B, float *__restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N * N)
    return;

  // 一次性加载到寄存器，消除重复全局访问
  float aval = A[idx];
  float bval = B[idx];
  float res = 0.0f;

  // 密集计算，大幅提高 FLOP/Byte 算术强度
  for (int i = 0; i < 32; ++i) {
    res += aval * bval + aval * 1.2f - bval * 0.8f;
    res = sin(res) + cos(res);
    res *= aval + bval;
  }
  C[idx] = res;
}

void checkCuda(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    exit(1);
  }
}

int main() {
  size_t size = N * N * sizeof(float);
  float *hA, *hB, *hC;
  float *dA, *dB, *dC;

  hA = (float *)malloc(size);
  hB = (float *)malloc(size);
  hC = (float *)malloc(size);

  for (int i = 0; i < N * N; ++i) {
    hA[i] = rand() / 1000.0f;
    hB[i] = rand() / 1000.0f;
  }

  checkCuda(cudaMalloc(&dA, size));
  checkCuda(cudaMalloc(&dB, size));
  checkCuda(cudaMalloc(&dC, size));

  checkCuda(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

  dim3 block(BLOCK);
  dim3 grid((N * N + BLOCK - 1) / BLOCK);

  // 1. 执行低效核函数
  naiveKernel<<<grid, block>>>(dA, dB, dC);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  // 2. 执行优化核函数
  optKernel<<<grid, block>>>(dA, dB, dC);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(hA);
  free(hB);
  free(hC);

  return 0;
}
