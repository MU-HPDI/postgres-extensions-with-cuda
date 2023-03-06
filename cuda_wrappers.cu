#include "cuda_kernel.hpp"

void cuda_wrapper_vector_addition(int *x, int *y, int *result, int n){
    // Set device to get it warmed up before we need it
    cudaSetDevice(0);

    int threads_per_block = 10;
    int no_of_blocks = n / threads_per_block;


    int *x_d, *y_d, *result_d;
    cudaMalloc((void **)&x_d, n*sizeof(int));
    cudaMalloc((void **)&y_d, n*sizeof(int));
    cudaMalloc((void **)&result_d, n*sizeof(int));
    
    cudaMemcpy(x_d, x, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, n*sizeof(int), cudaMemcpyHostToDevice);
    //  === CUDA ===
    vector_add_cuda<<<no_of_blocks,threads_per_block>>>(x_d, y_d, result_d, n);

    cudaMemcpy(result, result_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(x_d); cudaFree(y_d); cudaFree(result_d);
}
