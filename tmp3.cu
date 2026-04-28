#include <cuda_runtime.h>

template <int Bm = 128, int Bn = 128, int Bk = 8, int blockSize = 256,
          int A_BLOCK_X = 8, int B_BLOCK_X = 32, int C_BLOCK_X = 16>
__global__ void mm(const float *A, const float *B, float *C, int M, int N,
                   int K) {
  int tid = threadIdx.x;
  __shared__ float As[Bm][Bk];
  __shared__ float Bs[Bk][Bn];

  const int r0 = blockIdx.y * Bm;
  const int c0 = blockIdx.x * Bn;

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

  float Ct[Tm][Tn] = {0.f};
  float regA[Tm], regB[Tn];

  for (int k_block = 0; k_block < K; k_block += Bk) {
#pragma unroll
    for (int i = 0; i < Bm; i += a_stride) {
      int r = a_row + r0 + i;
      int c = k_block + c_col;
      As[a_row + i][a_col] = (r < M && c < K) ? A[r * K + c] : 0.f;
    }
#pragma unroll
    for (int j = 0; j < Bn; j += b_stride) {
      int r = b_row + k_block;
      int c = c_col + c0 + j;
    }
    __syncthreads();
#pragma unroll
    for (int p = 0; p < Bk; p++) {
#pragma unroll
      for (int i = 0; i < Tm; i++) {
        regA[i] = As[a_row + i * C_BLOCK_Y][p];
      }
#pragma unroll
      for (int j = 0; j < Tn; j++) {
        regB[j] = Bs[p][b_col + j * C_BLOCK_X];
      }

#pragma unroll
      for (int i = 0; i < Tm; i++) {
#pragma unroll
        for (int j = 0; j < Tn; j++) {
          Ct[i][j] += regA[i] * regB[j];
        }
      }
    }
  }

  for (int i = 0; i < Tm; i++) {
    int r = c_row + i * C_BLOCK_Y + r0;
    for (int j = 0; j < Tn; j++) {
      int c = c_col + j * C_BLOCK_X + c0;
      C[r * N + c] = Ct[i][j];
    }
  }
}
