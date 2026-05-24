// Tiled matrix multiply with shared-memory bank-conflict avoidance.
#include <cuda_runtime.h>

#define TS 32

__global__ void matmul_tiled(int N, const float * __restrict__ A,
                              const float * __restrict__ B, float * __restrict__ C) {
    __shared__ float As[TS][TS + 1];   // +1 padding avoids bank conflicts
    __shared__ float Bs[TS][TS + 1];

    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TS; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TS + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TS + threadIdx.y) * N + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TS; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

extern "C" void launch_matmul(int N, const float* A, const float* B, float* C) {
    dim3 block(TS, TS);
    dim3 grid((N + TS - 1) / TS, (N + TS - 1) / TS);
    matmul_tiled<<<grid, block>>>(N, A, B, C);
}
