#include "cuda_kernel.cu"

#include <numeric>

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

void cuda_wrapper_heart_rate_estimation(
    std::vector<std::vector<unsigned short int>> selected_filter_vector,
    float *heart_rate,
    int heart_rate_size,
    int num_of_threads,
    float *time_elapsed
)
{

    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda, 0);

    thrust::host_vector<int> h_offset_vector(heart_rate_size + 1);
    int num_of_elements = 0;

    for (int i = 0; i < heart_rate_size; i++)
    {
        std::vector<unsigned short int> temp = selected_filter_vector[i];
        h_offset_vector[i] = num_of_elements;
        num_of_elements += selected_filter_vector[i].size();
    }
    h_offset_vector[heart_rate_size] = num_of_elements;

    // Step 1: Flatten the vector of vectors
    std::vector<unsigned short int> flattened_vec;
    flattened_vec.reserve(std::accumulate(selected_filter_vector.begin(), selected_filter_vector.end(), 0,
        [](size_t acc, const std::vector<unsigned short int>& vec) {
            return acc + vec.size();
        }));

    for (const auto& vec : selected_filter_vector) {
        flattened_vec.insert(flattened_vec.end(), vec.begin(), vec.end());
    }

    // Step 2: Convert the flattened vector to a device_vector
    thrust::device_vector<unsigned short int> d_selected_filter_vector(flattened_vec.begin(), flattened_vec.end());

    typedef double T;

    thrust::device_vector<int> d_offset_vector = h_offset_vector;

    thrust::device_vector<T> d_heart_rate(heart_rate_size);

    unsigned short int *d_selected_filter_vector_ptr = thrust::raw_pointer_cast(d_selected_filter_vector.data());
    int *d_offset_vector_ptr = thrust::raw_pointer_cast(d_offset_vector.data());
    T *d_heart_rate_ptr = thrust::raw_pointer_cast(d_heart_rate.data());

    int blocks = heart_rate_size;
    int array_size = 6'000;
    int slide = 500;

    thrust::device_vector<T> d_global_output_with_padding(heart_rate_size * (array_size + slide));
    T *d_global_output_with_padding_ptr = thrust::raw_pointer_cast(d_global_output_with_padding.data()); 

    hr_kernel<T, 6'000, 500> <<<blocks, num_of_threads >>>(
        d_selected_filter_vector_ptr,
        d_offset_vector_ptr,
        d_global_output_with_padding_ptr,
        d_heart_rate_ptr
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        std::cerr << "Error: " << cudaGetErrorString(err) <<  " Name: " << cudaGetErrorName(err) << std::endl;
    }

    thrust::copy(d_heart_rate.begin(), d_heart_rate.end(), heart_rate);


    cudaEventRecord(stop_cuda, 0);
    cudaEventSynchronize(stop_cuda);

    cudaEventElapsedTime(time_elapsed, start_cuda, stop_cuda);

}