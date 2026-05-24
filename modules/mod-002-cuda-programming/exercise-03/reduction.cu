// Three reduction variants: naive, shared-memory, warp-shuffle.
#include <cuda_runtime.h>

__global__ void reduce_naive(int n, const float* in, float* out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];  // BAD: divergent
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__global__ void reduce_shared(int n, const float* in, float* out) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];  // sequential addressing
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

__inline__ __device__ float warp_reduce(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    return v;
}

__global__ void reduce_shfl(int n, const float* in, float* out) {
    float sum = 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sum = in[i];
    sum = warp_reduce(sum);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / 32) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce(sum);
        if (threadIdx.x == 0) out[blockIdx.x] = sum;
    }
}
