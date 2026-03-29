#include <cuda_runtime.h>

template <int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256,
          int A_BLOCK_X = 8, int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C,
                                             int M, int K, int N)
{
    // 共享内存
    __shared__ float As[Bm][Bk];
    __shared__ float Bs[Bk][Bn];

    const int r0 = blockIdx.y * Bm;
    const int c0 = blockIdx.x * Bn;
    const int tid = threadIdx.x;

    // ===================== 加载 A 的索引 =====================
    // 256线程加载 128*8 元素 -> 每人 load 4 个
    const int a_row = tid / A_BLOCK_X;          // tid / 8 -> 0~31
    const int a_col = tid % A_BLOCK_X;          // tid % 8 -> 0~7
    const int a_stride = blockSize / A_BLOCK_X; // 32

    // ===================== 加载 B 的索引 =====================
    // 256线程加载 8*128 元素 -> 每人 load 4 个
    const int b_row = tid / B_BLOCK_X; // tid / 32 -> 0~7
    const int b_col = tid % B_BLOCK_X; // tid % 32 -> 0~31
    const int b_stride = B_BLOCK_X;    // 32 (沿列方向跳转)

    // ===================== 计算 C 的映射 =====================
    constexpr int C_BLOCK_Y = blockSize / C_BLOCK_X; // 256 / 16 = 16
    const int c_row = tid / C_BLOCK_X;               // 0~15
    const int c_col = tid % C_BLOCK_X;               // 0~15

    constexpr int Tm = Bm / C_BLOCK_Y; // 128 / 16 = 8
    constexpr int Tn = Bn / C_BLOCK_X; // 128 / 16 = 8

    float Ct[Tm][Tn];
#pragma unroll
    for (int i = 0; i < Tm; i++)
    {
#pragma unroll
        for (int j = 0; j < Tn; j++)
        {
            Ct[i][j] = 0.0f;
        }
    }
    float regA[Tm];
    float regB[Tn];
    // ===================== K 维度大循环 =====================
    for (int k_block = 0; k_block < K; k_block += Bk)
    {

// 加载 A (纵向平铺)
#pragma unroll
        for (int i = 0; i < Bm; i += a_stride)
        {
            int r = r0 + a_row + i;
            int c = k_block + a_col;
            As[a_row + i][a_col] = (r < M && c < K) ? A[r * K + c] : 0.0f;
        }

// 加载 B (横向平铺 - 关键修正点)
#pragma unroll
        for (int j = 0; j < Bn; j += b_stride)
        {
            int r = k_block + b_row;
            int c = c0 + b_col + j;
            Bs[b_row][b_col + j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        __syncthreads();

// ===================== 计算核心 =====================
#pragma unroll
        for (int p = 0; p < Bk; p++)
        {
            // [x] 新增 reg
#pragma unroll
            for (int i = 0; i < Tm; i++)
            {
                regA[i] = As[c_row + i * C_BLOCK_Y][p];
            }
#pragma unroll
            for (int j = 0; j < Tn; j++)
            {
                regB[j] = Bs[p][c_col + j * C_BLOCK_X];
            }
#pragma unroll
            for (int i = 0; i < Tm; i++)
            {
#pragma unroll
                for (int j = 0; j < Tn; j++)
                {
                    // Ct 存储的，一个线程处理的 64 个数，不是连续的
                    Ct[i][j] += As[c_row + i * C_BLOCK_Y][p] * Bs[p][c_col + j * C_BLOCK_X];
                }
            }
        }

        __syncthreads();
    }

// ===================== 写回结果 =====================
#pragma unroll
    for (int i = 0; i < Tm; i++)
    {
        int r = r0 + c_row + i * C_BLOCK_Y;
#pragma unroll
        for (int j = 0; j < Tn; j++)
        {
            int c = c0 + c_col + j * C_BLOCK_X;
            if (r < M && c < N)
            {
                C[r * N + c] = Ct[i][j];
            }
        }
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int K, int N)
{
    dim3 threadsPerBlock(256); // 对应 blockSize
    dim3 blocksPerGrid((N + 128 - 1) / 128, (M + 128 - 1) / 128);
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
