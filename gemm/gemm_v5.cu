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
    int tAx = tidx & (A_BLOCK_X - 1);
    int tAy = tidx / A_BLOCK_X;
    int tBx = tidx & (B_BLOCK_X - 1);
    int tBy = tidx / B_BLOCK_X;

    int warpId = tidx >> 5;
    int laneId = tidx & 31;
    int warpx = warpId & (C_WARP_DIM - 1);
    int warpy = warpId / C_WARP_DIM;
    int lanex = (laneId & 15) >> 1;
    int laney = ((laneId >> 4) << 1) + (laneId & 1);
    /**
     * [
     *  [0, 1,  2,  3,  4,  5,  6,  7]
     *  [8, 9,  10, 11, 12, 13, 14, 15]
     * ]
     * - - - >
     * [
     *  [0, 2, 4, 6, 8...]
     *  [1, 3, 5, 7, 9...]
     * ]
     */
    int tCx = warpx * C_WARP_X + lanex;
    int tCy = warpy * C_WARP_Y + laney;
    float acc[Tm][Tn] = {0.0f};
    float regA[2][Tm];
    float regB[2][Tn];
    int bufferId = 0;
    // [x] 双循环预缓冲 0
#pragma unroll
    for (int i = 0; i < Bm; i += A_BLOCK_Y)
    {
        int r = row0 + i + tAy;
        // [x] 这里的异或 tAx 0~8 <<2 0, 0, 0, 0; 4, 4, 4, 4 ... 按照 tAx 变换，每 4 个位置循环一次的乱序共享内存索引
        // tAx = 0,不交换 tAx=1，四个为一组上下交换 (0-4,1-5,2-6,3-7) tAx=2,八个一组上下交换...依此类推，一共一次是 8 列，128 行 所以 tAx=8 的时候上下交换 32 为一组
        sA[0][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && tAx < K) ? A[r * K + tAx] : 0.0f;
    }
#pragma unroll

#pragma unroll
    for (int j = 0; j < Bn; j += B_BLOCK_X)
    {

        int r = tBy;
        int c = col0 + j + tBx;
        sB[0][r][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
    }

    __syncthreads();

    for (int k = Bk; k < K + Bk; k += Bk)
    {
#pragma unroll
        for (int tk = 0; tk < Bk + 1; ++tk)
        {
            if (tk > 0) //  [x] 先算再读
            {
#pragma unroll
                for (int tm = 0; tm < Tm; tm++)
                {
#pragma unroll
                    for (int tn = 0; tn < Tn; tn++)
                    {
                        acc[tm][tn] += regA[(tk - 1) & 1][tm] * regB[(tk - 1) & 1][tn];
                    }
                }
            }
            if (tk < Bk)
            {
#pragma unroll
                for (int tm = 0; tm < Tm >> 2; tm++)
                {
                    int r = (tCy + tm * C_BLOCK_Y) << 2;
                    FLOAT4(regA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
                }
#pragma unroll
                for (int tn = 0; tn < Tn >> 2; tn++)
                {
                    int c = (tCx + tn * C_BLOCK_X) << 2;
                    FLOAT4(regB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
                }
            }
        }

        if (k < K)
        {
            int c = k + tAx;
#pragma unroll
            for (int i = 0; i < Bm; i += A_BLOCK_Y)
            {
                int r = row0 + i + tAy;
                sA[bufferId ^ 1][tAx][(i + tAy) ^ ((tAx) << 2)] = (r < M && c < K) ? A[r * K + c] : 0.0f;
            }
#pragma unroll

            int r = k + 0 + tBy;
#pragma unroll
            for (int j = 0; j < Bn; j += B_BLOCK_X)
            {
                int c = col0 + j + tBx;
                sB[bufferId ^ 1][0 + tBy][j + tBx] = (r < K && c < N) ? B[r * N + c] : 0.0f;
            }

            __syncthreads();
        }
        bufferId ^= 1;
    }
#pragma unroll
    for (int i = 0; i < Tm; i++)
    {
        int r = row0 + (tCy << 2) + (i & (~3)) * C_BLOCK_Y + (i & 3);
#pragma unroll
        for (int j = 0; j < Tn; j++)
        {
            int c = col0 + (tCx << 2) + (j & (~3)) * C_BLOCK_X + (j & 3);
            if (r < M && c < N)
                C[r * N + c] = acc[i][j];
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