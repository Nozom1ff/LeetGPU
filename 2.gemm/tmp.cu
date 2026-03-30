#include <cuda_runtime.h>

#define FLOAT4(x) (reinterpret_cast<float4 *>(&(x))[0])

constexpr int Bm = 128;
constexpr int Bn = 128;
constexpr int Bk = 8;
constexpr int BLOCK_SIZE = 256;
constexpr int A_BLOCK_X = 8;
constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;
constexpr int B_BLOCK_X = 32;
constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;
constexpr int C_BLOCK_X = 16;
constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;
constexpr int Tn = Bn / C_BLOCK_Y;
constexpr int Tm = Bm / C_BLOCK_Y;
constexpr int C_WARP_X = 8;
constexpr int C_WARP_Y = 32 / C_WARP_X;
constexpr int C_WARP_DIM = C_BLOCK_X / C_WARP_X;

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    int col0 = blockIdx.x * Bn;
    int row0 = blockIdx.y * Bm;
    int tidx = threadIdx.x;
    __shared__ float sA[2][Bk][Bm];
    __shared__ float sB[2][Bk][Bn];
    int tAx = tid & (A_BLOCK_X - 1);
    int tAy = tid / A_BLOCK_X;
    int tBx = tid & (B_BLOCK_X - 1);
    int tBy = tid / (B_BLOCK_X);

    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (C_WARP_DIM - 1);
    int warpy = warpId / C_WARP_DIM;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);

    int tCx = warpx * C_WARP_X + lanex;
    int tCy = warpy * C_WARP_Y + laney;
    float acc[Tm][Tn] = {0.0f};
    float regA[2][Tm];
    float regB[2][Tn];
    int bufferId = 0;
    // pre
#pragma unroll
    for (int i = 0; i < Bm; i += A_BLOCK_Y)
    {
        int r = row0 + i + tAy;
        sA[0][tAx][(i + tAy) ^ (tAx << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < Bn; j += B_BLOCK_X)
    {
        int c = col0 + j + tBx;
        sB[0][tBy][j + tBx] = (tBy < K && c < N) ? B[r * N + c] : 0.0f;
    }
    __syncthreads();
    for (int k = Bk; k < K + Bk; k += Bk)
    {
#pragma unroll
        for (int tk = 0; tk < Bk + 1; ++tk)
        {
            if (tk > 0)
            {
#pragma unroll
                for (int tm = 0; tm < Tm; tm++)
                {
#pragma unroll
                    for (int tn = 0; tn < Tn; tn++)
                    {
                        acc[tm][tn] = regA[(tk - 1) & 1][tm] * regB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk != Bk)
            {
#pragma unroll
                for (int tm = 0; tm < Tm >> 2; tm++)
                {
                    int r = (tCy + tm * C_BLOCK_Y) << 2;
                    FLOAT4(regA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
#pragma
                for (int tn = 0; tn < Tn >> 2; tn++)
                {
                    int c = (tCx + tn * C_BLOCK_X) << 2;
                    FLOAT4(regB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 blocksPerGrid((K + Bn - 1) / Bn,
                       (M + Bm - 1) / Bm);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
