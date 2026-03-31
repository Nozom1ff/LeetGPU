#include <cuda_runtime.h>
template <int BLOCK_SIZE>
__global__ void reduce_kernel(const float *input, float *output, int N) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  sdata[tid] = i < N ? input[i] : 0.0f;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    output[blockIdx.x] = sdata[0]; // 只用 thread0 写
  }
}

extern "C" void solve(const float *d_input, float *d_output, int N) {
  const int BLOCK_SIZE = 256;
  int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  float *temp; // 临时存每个 block 的部分和
  cudaMalloc(&d_block_sum, grid_size * sizeof(float)); // 记得 cudaMalloc

  // 第一轮 和后面的轮次不一样
  reduce_kernel<256><<<grid_size, BLOCK_SIZE>>>(d_input, temp, N);
  // 第二轮 reduce，把 block 结果不断规约为 1 个数
  int remaining = grid_size;
  while (remaining > 1) {
    int new_grid = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_kernel<256><<<new_grid, BLOCK_SIZE>>>（temp, temp, remaining）;
    remaining = new_grid;
  }
  // 最终结果 copy
  // 最终结果 copy 到 d_output
  cudaMemcpy(d_output, temp, sizeof(float), cudaMemcpyDeviceToDevice);

  cudaFree(temp);
}
// - - - - - - - - - 上面是baseline

#define scale 8 // 一个线程处理八个元素
#define BLOCK_SIZE 256

__global__ void reduction(const float *input, float *output, int N) {
  __shared__ float smem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int block_offset = blockDim.x * blockIdx.x * scale;
  float sum = 0.0;
  for (int step = 0; step < scale; ++step) {
    // 注意访问连续性
    int idx = block_offset + (step * blockDim.x) + tid;
    sum += (idx < N) ? input[idx] : 0.0f;
  }
  smem[tid]=sum;
  __syncthreads();
  // 手动展开
   if(BLOCK_SIZE>=512&&tid<256){
	smem[tid]+=smem[tid+256];
	__syncthreads();
   }
   if(BLOCK_SIZE>=256&&tid<128){
	smem[tid]+=smem[tid+128];
	__syncthreads();
   }
   if(BLOCK_SIZE>=128&&tid<64){
	smem[tid]+=smem[tid+64];
	__syncthreads();
   }
   // 小于 32
   if(tid<32){
		float val = smem[tid] + smem[tid+32];
		// 32 内 直接 warp shuffle
		val += __shfl_down_sync(0xffffffff,val,16);
		val += __shfl_down_sync(0xffffffff,val,8);
		val += __shfl_down_sync(0xffffffff,val,4);
		val += __shfl_down_sync(0xffffffff,val,2);
		val += __shfl_down_sync(0xffffffff,val,1);
		// 直接写入A0!
		if(tid==0){
			atomicAdd(output, val);
		}
   }
}
// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int block_size = BLOCKSIZE;
    int new_block_capacity = block_size * add_during_load_scaling;
    int grid_size = (N + new_block_capacity - 1) / new_block_capacity;
    // size_t smem_size = block_size * sizeof(float);

    // reduct_kernel<<<grid_size, block_size, smem_size>>> (input, output, N);
    reduct_kernel<<<grid_size, block_size>>> (input, output, N);
}









