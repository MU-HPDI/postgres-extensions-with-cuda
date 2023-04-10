#ifndef CUDA_KERNEL_H /*/ Include guard */
#define CUDA_KERNEL_H

#define BLOCK_SIZE 1024

__global__ void vector_add_cuda(int *x, int *y, int *result, int n);
__global__ void max_kernel(short int *input, short int *output, int rows, int cols);
#endif // CUDA_KERNEL_H