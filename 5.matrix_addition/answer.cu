#include <cuda_runtime.h>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int by = 16*blockIdx.y+threadIdx.y;
    int bx = 16*blockIdx.x+threadIdx.x;
    if(by<N&&bx<N){
        C[by*N+bx]=A[by*N+bx]+B[by*N+bx];
    }

}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    dim3 threadsPerBlock = dim3(16,16);
    dim3 blocksPerGrid = dim3((N+16-1)/16,(N+16-1)/16);

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
/* - - - - - - - - 上面是我的版本 - - - - - - - - - - */
/**
 * 优化版本，不要分 ty,tx 统一 tx，方便地址的顺序访问
 *  统一看作一位数组处理
 */
__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    const auto tid = threadIdx.x;
    const auto id = blockDim.x * blockIdx.x + tid;

    if (id >= N) {
        return;
    }
    C[id] = A[id] + B[id];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N + threadsPerBlock - 1) / threadsPerBlock;

    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N*N);
    // cudaDeviceSynchronize();
}


