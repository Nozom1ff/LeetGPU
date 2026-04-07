#include <cmath>
#include <cuda_runtime.h>
#include <iostream>

// 设定超参数，对应推荐的小尺寸块
#define d 32
#define Br 32
#define Bc 32

__global__ void flash_attn_v1_kernel(const float *Q, const float *K,
                                     const float *V, float *O, int N) {
  // 当前 Block 处理 Q 的第 i 个块，那么一个线程负责一行，只存一个 m，一个 l
  int i_idx = blockIdx.x;
  int tid = threadIdx.x;
  int global_i = i_idx * Br + tid;
  __shared__ float s_K[Bc][d];
  __shared__ float s_V[Bc][d]; // 块内 kv 串行 块外 q 并行
                               // 下面的变量都放在寄存器中
  float t_Q[d];
  float t_O[d] = {0.0f};
  float t_m = -1e20f;
  float t_l = 0.0f;
  // 写入 Q_i
  if (global_i < N) {
    for (int k = 0; k < d; ++k) {
      t_Q[k] = Q[global_i * d + k];
    }
  }
  // 计算 kv 有多少 tile
  int Tc = (N + Bc - 1) / Bc;
  // 外层循环，遍历KV的所有块
  for (int j = 0; j < Tc; ++j) {

    int global_j = j * Bc + tid;
    if (global_j < N) {
      for (int k = 0; k < d; ++k) {
        s_K[tid][k] = K[global_j * d + k];
        s_V[tid][k] = V[global_j * d + k];
      }
    }
    __syncthreads();

    // 边界保护
    if (global_i < N) {
      // 保存当前 query 和 k_j 算出来的 score
      float S_ij[Bc]; // 注意，每次到一个新的 kv tile 就初始化这个数组
                      // 这个不需要长期保存
      float m_ij = -1e20f;
      for (int k = 0; k < Bc; ++k) {
        float sum = 0.0f;
        for (int x = 0; x < d; ++x)
          sum += t_Q[x] * s_K[k][x];
        S_ij[k] = sum;
        m_ij = fmaxf(m_ij, sum); // 更新 m
      }
      // 计算 m_new 和 l_new
      float m_new = fmaxf(t_m, m_ij);
      float l_ij = 0.0f;
      float P_ij[Bc];

      float(int k = 0; k < Bc; ++k) {
        if (j * Bc + k < N) {
          P_ij[k] = expf(S_ij[k] - m_new);
          l_ij += P_ij[k];
        } else {
          P_ij[k] = 0.0f;
        }
      }
      float l_new = expf(t_m - m_new) * t_l + l_ij; // 每一块更新一次
      // 更新 O_i
      // O_i = (l_i*exp(m_i-m_new)*O_i+P_ij*V_j)
      for (int x = 0; x < d; ++x) {
        float pv = 0.0f;
        for (int k = 0; k < Bc; ++k)
          pv += P_ij[k] * s_V[k][x]; // 1xBC BCx1
        t_O[x] = (t_l * expf(t_m - m_new) * t_O[x] + pv) / l_new;
      }
      t_m = m_new;
      t_l = l_new;
    }
    __syncthreads();
  }
  if (global_i < N) {
    for (int k = 0; k < d; ++k) {
      O[global_i * d + k] = t_O[k];
    }
  }
}

void run_flash_attn_v1(const float* d_Q, const float* d_K, const float* d_V, float* d_O, int N) {
    // 设定 Grid 和 Block 维度
    // Block 大小为 Br(32)，对应每个 Block 处理 Q 的 32 行
    int threads_per_block = Br;
    // Grid 大小对应 T_r = N / Br
    int blocks_per_grid = (N + Br - 1) / Br;

    flash_attn_v1_kernel<<<blocks_per_grid, threads_per_block>>>(d_Q, d_K, d_V, d_O, N);
}
