#ifndef CUDA_KERNEL_H /*/ Include guard */
#define CUDA_KERNEL_H

__global__ void vector_add_cuda(int *x, int *y, int *result, int n);

#endif // CUDA_KERNEL_H