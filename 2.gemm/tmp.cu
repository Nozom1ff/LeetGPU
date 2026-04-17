#include <cuda_runtime.h>

template <int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256, int A_BLOCK_X = 8, int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void gemm(const float *A, const float *B, float *C, int M, int K, int K)
{
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    const int r0 = blockIdx.y * Bm;
    const int c0 = blockIdx.x * Bn;
    const int tid = threadIdx.x;

    const int a_row = tid / A_BLOCK_X;
    const int a_col = tid % A_BLOCK_X;
    const int a_stride = blockSize / A_BLOCK_X;
    const int b_row = tid / B_BLOCK_X;
    const int b_col = tid % B_BLOCK_X;
    const int b_stride = blockSize / B_BLOCK_X;

    constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X;
    const int c_row = tid / C_BLOCK_X;
    const int c_col = tid % C_BLOCK_X;

    constexpr int Tm = Bm / C_BLOCK_Y;
    constexpr int Tn = Bn / C_BLOCK_X;

    float Ct[Tm][Tn];
#pragma unroll
    for (int i = 0; i < Tm; i++)
#pragma unroll
        for (int j = 0; j < Tn; j++)
            Ct[i][j] = 0.0f;

    for (int k_block = 0; k_block < K; k_block += Bk)
    {
#pragma unroll
        for (int i = 0; i < Bm; i += a_stride)
        {
            int r = r0 + i + a_row;
            int c = k_block + a_col;
            As[a_row + i][a_col] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < Bn; j += b_stride)
        {
            int r = k_block + b_row;
            int c = c0 + j + b_col;
            Bs[b_row][b_col + j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int p = 0; p < Bk; p++)
        {
            for (int i = 0; i < Tm; i++)
            {
                for (int j = 0; j < Tn; j++)
                {
                    Ct[i][j] += As[c_row + i * C_BLOCK_Y][p] * Bs[p][c_col + j * C_BLOCK_X];
                }
            }
        }
    }
}