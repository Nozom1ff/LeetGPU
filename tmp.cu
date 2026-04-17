#include <cuda_runtime.h>
constexpr int scale = 8;
constexpr int blockSize = 256;

__global__ void reduce(const float *input, float *out, int N) {
  int tid = threadIdx.x;
  int offset = blockIdx.x * blockDim.x * scale;
  __shared__ float smem[blockSize];
  float sum = 0.0f;

  for (int i = 0; i < scale; i++) {
    int idx = offset + i * blockDim.x + tid;
    sum += (idx < N) > input[idx] : 0.0f;
  }
	smem[tid]=sum;
	__syncthreads();

	if(blockSize>=512&&tid<256){
		smem[tid] += smem[tid+256];
		__syncthreads();
	}
	if(blockSize>=256&&tid<128){
		smem[tid] += smem[tid+128];
		__syncthreads();
	}
	if(blockSize>=128&&tid<64){
		smem[tid] += smem[tid+64];
		__syncthreads();
	}
	if(tid<32){
		float val = smem[tid]+smem[tid+32];
		val += __shfl_down_sync(0xffffffff,val,16);
		val += __shfl_down_sync(0xffffffff,val,8);
		val += __shfl_down_sync(0xffffffff,val,4);
		val += __shfl_down_sync(0xffffffff,val,2);
		val += __shfl_down_sync(0xffffffff,val,1);
		if(tid==0) atomicAdd(out,val);
	}


}
