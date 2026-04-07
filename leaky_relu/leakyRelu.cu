// 我的版本 base 0.30ms
// #include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if(tid<N){
        float x = input[tid];
        output[tid] = x>0?x:0.01*x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}

// 优化版，向量化访存 0.26ms
#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* input, float* output, int N)
{
    // 一个线程负责一个float4 vector 对应到地址的话需要4倍
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * 4;
    if(offset >= N) return ;

    if(offset + 3 < N){
        // 正常操作即可
        const float4* in_ptr = reinterpret_cast<const float4*>(&(input[offset]));
        float4* out_ptr = reinterpret_cast<float4*>(&(output[offset]));

        float4 val = *in_ptr;
        val.x = val.x > 0.0f ? val.x : 0.01f * val.x;
        val.y = val.y > 0.0f ? val.y : 0.01f * val.y;
        val.z = val.z > 0.0f ? val.z : 0.01f * val.z;
        val.w = val.w > 0.0f ? val.w : 0.01f * val.w;

        *out_ptr = val;
    }
    else if(offset < N){
        // 注意for循环范围
        for(int i = offset; i < N; i++){
            float temp = input[i];
            output[i] = temp > 0.0f ? temp : 0.01f * temp;
        }
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 128;
    int n_vec = (N + 3) / 4; // float4向量块的个数
    int blocksPerGrid = (n_vec + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
// NOTE!!! 补充说明 
// __expf = 快 + 精度低；expf = 标准 + 精度高 __expf 是 nvidia 内置函数
// 注意 ！
//
