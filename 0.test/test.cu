#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// ==============================================
// 这是 CUDA 官方标准错误检查宏（绝对正确）
// ==============================================
#define CHECK_CUDA_ERROR(err)                                                \
  do {                                                                       \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), \
              __LINE__);                                                     \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

__global__ void vectorFunc(float* arr, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) {
    arr[idx] = (float)idx * 2.0f;
  }
}

int main() {
  const int arraySize = 2048;
  float* host_array = nullptr;
  float* dev_array = nullptr;

  // 分配主机内存
  host_array = (float*)malloc(arraySize * sizeof(float));
  for (int i = 0; i < arraySize; i++) {
    host_array[i] = 1.0f;
  }

  // 分配设备内存
  CHECK_CUDA_ERROR(cudaMalloc((void**)&dev_array, arraySize * sizeof(float)));

  // 拷贝主机 -> 设备
  CHECK_CUDA_ERROR(cudaMemcpy(dev_array, host_array, arraySize * sizeof(float),
                              cudaMemcpyHostToDevice));

  // 启动核函数（使用 256 线程，兼容所有显卡）
  dim3 blockSize(256);
  dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x);

  vectorFunc<<<gridSize, blockSize>>>(dev_array, arraySize);

  // 核函数后：先检查启动错误，再同步
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  // 拷贝回主机
  CHECK_CUDA_ERROR(cudaMemcpy(host_array, dev_array, arraySize * sizeof(float),
                              cudaMemcpyDeviceToHost));

  // 打印结果
  printf("===== 结果 =====\n");
  for (int i = 0; i < 10; i++) {
    printf("%.2f\n", host_array[i]);
  }

  // 释放
  CHECK_CUDA_ERROR(cudaFree(dev_array));
  free(host_array);

  printf("执行完毕\n");
  return 0;
}
