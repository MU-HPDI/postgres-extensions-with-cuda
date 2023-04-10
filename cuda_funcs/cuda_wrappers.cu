#include "cuda_kernel.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/iterator_facade.h>
#define BLOCK_SIZE 1024

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


void cuda_find_max(const short int* data, short int *output, int rows, int cols) 
{

    thrust::device_vector<short int> d_data(rows * cols);
    thrust::copy(data, data + (rows * cols), d_data.begin());

    short int *d_data_ptr = thrust::raw_pointer_cast(d_data.data());

    dim3 blockSize(BLOCK_SIZE, 1, 1);
    dim3 gridSize(ceil((float)cols / BLOCK_SIZE), rows, 1);

    thrust::device_vector<short int> d_result(rows);
    short int *d_result_ptr = thrust::raw_pointer_cast(d_result.data());

    max_kernel<<<gridSize, blockSize>>>(d_data_ptr, d_result_ptr, rows, cols);
    cudaDeviceSynchronize();

    // get last CUDA error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    thrust::host_vector<short int> h_result = d_result;

    // copy h_result to output
    for (int i = 0; i < rows; i++)
    {
        output[i] = h_result[i];
    }

}