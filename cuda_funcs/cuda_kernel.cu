#include <math.h>
#include <stdio.h>
#include "cuda_kernel.hpp"


__global__
void vector_add_cuda(int *x, int *y, int * result, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  result[i] = x[i] + y[i];
}

__global__ void max_kernel(short int* input, short int* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    __shared__ short int sdata[BLOCK_SIZE];
    
    if (row < rows && col < cols) {

        // Initialize the shared memory with appropriate input value
        sdata[threadIdx.x] = (threadIdx.x < cols) ? input[idx] : INT_MIN;
        __syncthreads();

        // Perform sequential addressing reduction to find the max value in each row
        for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride && threadIdx.x + stride < cols) {
                short int lhs = sdata[threadIdx.x];
                short int rhs = sdata[threadIdx.x + stride];
                sdata[threadIdx.x] = (lhs < rhs) ? rhs : lhs;
            }
            __syncthreads();
        }
    }
    
    // Store the max value of each row in the output array
    if (threadIdx.x == 0 && row < rows) {
        output[row] = sdata[0];
    }
}