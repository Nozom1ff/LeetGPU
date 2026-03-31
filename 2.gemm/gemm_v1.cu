/**
 * base + shared_mem
 * baseline 923.37ms
 * current version:717.9 ms
 */
#include <cuda_runtime.h>

constexpr int TILE = 16;

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
	__shared__ float ma[TILE][TILE];
	__shared__ float mb[TILE][TILE];
	int tx = threadIdx.x, ty = threadIdx.y;
	int row0 = blockIdx.y * blockDim.y + ty, col0 = blockIdx.x * blockDim.x + tx;
	int n_tile = (K + TILE - 1) / TILE;
	float sum = 0.f;
#pragma unroll
	for (int p = 0; p < n_tile; p++)
	{
		int px = p * TILE + tx;
		int py = p * TILE + ty;
		ma[ty][tx] = (row0 < M && px < K) ? A[row * K + px] : 0.f;
		mb[ty][tx] = (py < K && col0 < N) ? B[py * N + col0] : 0.f;
		__syncthreads();
		for (int i = 0; i < TILE; i++)
		{
			sum += ma[ty][i] * mb[i][tx];
		}
		__syncthreads();
	}
	if (row0 < M && col0 < K)
	{
		C[row0 * N + col0] = sum;
	}
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, const float *B, float *C, int M, int K, int N)
{
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
					   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

	matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
}

