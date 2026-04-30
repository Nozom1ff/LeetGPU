#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// 一个非常简单的 Kernel，计算开销极低，用来放大 CPU 提交开销的影响
__global__ void tiny_kernel(float *data, int idx) { data[idx] += 1.0f; }

const int NUM_KERNELS = 100; // 每个任务包含 100 个小 kernel
const int ITERATIONS = 1000; // 重复运行 1000 次进行计时对比

int main() {
  float *d_data;
  cudaMalloc(&d_data, NUM_KERNELS * sizeof(float));
  cudaMemset(d_data, 0, NUM_KERNELS * sizeof(float));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // --- 1. 传统流式执行 (Traditional Stream) ---
  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);

  cudaEventRecord(start1, stream);
  for (int i = 0; i < ITERATIONS; i++) {
    for (int j = 0; j < NUM_KERNELS; j++) {
      // 每次启动都会产生 CPU 侧的 Driver 开销
      tiny_kernel<<<1, 1, 0, stream>>>(d_data, j);
    }
  }
  cudaEventRecord(stop1, stream);
  cudaEventSynchronize(stop1);
  float ms_stream = 0;
  cudaEventElapsedTime(&ms_stream, start1, stop1);

  // --- 2. CUDA Graph 执行 ---
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  // A. 捕获图 (Capture)
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  for (int j = 0; j < NUM_KERNELS; j++) {
    tiny_kernel<<<1, 1, 0, stream>>>(d_data, j);
  }
  cudaStreamEndCapture(stream, &graph);

  // B. 实例化图 (Instantiation)
  // 这一步会进行预处理，优化调度路径
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  // C. 执行图 (Execution)
  cudaEvent_t start2, stop2;
  cudaEventCreate(&start2);
  cudaEventCreate(&stop2);

  cudaEventRecord(start2, stream);
  for (int i = 0; i < ITERATIONS; i++) {
    // 一次启动包含 100 个 kernel，CPU 只需下达一个指令
    cudaGraphLaunch(graphExec, stream);
  }
  cudaEventRecord(stop2, stream);
  cudaEventSynchronize(stop2);
  float ms_graph = 0;
  cudaEventElapsedTime(&ms_graph, start2, stop2);

  // 打印结果
  std::cout << "Results for " << ITERATIONS << " runs of " << NUM_KERNELS
            << " kernels:" << std::endl;
  std::cout << "Traditional Stream Time: " << ms_stream << " ms" << std::endl;
  std::cout << "CUDA Graph Time:         " << ms_graph << " ms" << std::endl;
  std::cout << "Speedup:                 " << ms_stream / ms_graph << "x"
            << std::endl;

  // 清理
  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaFree(d_data);
  cudaStreamDestroy(stream);
  return 0;
}
