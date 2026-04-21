#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void dummy_math_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // 模拟计算负载
        for(int i = 0; i < 100; ++i) {
            data[idx] = data[idx] * 2.0f - 1.0f;
        }
    }
}

int main() {
    int size = 1 << 20;
    float *d_data;
    cudaMalloc(&d_data, size * sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // ==========================================
    // 方案一：CPU 计时方法 (std::chrono)
    // ==========================================
    
    // 强制设备同步，确保之前的状态清空
    cudaDeviceSynchronize(); 
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    
    // 异步内核下发
    dummy_math_kernel<<<blocks, threads>>>(d_data, size);
    
    // 【关键点】：必须在此处调用设备同步
    cudaDeviceSynchronize(); 
    
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_stop - cpu_start;
    std::cout << "CPU Timing: " << cpu_duration.count() << " ms\n";

    // ==========================================
    // 方案二：CUDA Event 计时方法
    // ==========================================
    
    cudaEvent_t event_start, event_stop;
    cudaEventCreate(&event_start);
    cudaEventCreate(&event_stop);

    // 将起始时间戳指令推入默认流的硬件队列
    cudaEventRecord(event_start, 0);
    
    // 异步内核下发
    dummy_math_kernel<<<blocks, threads>>>(d_data, size);
    
    // 将结束时间戳指令推入相同的硬件队列
    cudaEventRecord(event_stop, 0);
    
    // 阻塞 CPU，直到 stop 事件在 GPU 物理层面上被触发
    cudaEventSynchronize(event_stop);
    
    float gpu_duration = 0.0f;
    // 计算两个硬件时间戳之间的差值
    cudaEventElapsedTime(&gpu_duration, event_start, event_stop);
    std::cout << "CUDA Event Timing: " << gpu_duration << " ms\n";

    cudaEventDestroy(event_start);
    cudaEventDestroy(event_stop);
    cudaFree(d_data);
    return 0;
}
