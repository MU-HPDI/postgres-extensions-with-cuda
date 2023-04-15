#include <iostream>
#include <math.h>
#include <random>
#include <chrono>


#include "heart_rate_device_funcs.cu"

#define SLIDE 500
#define BLOCK_SIZE 1024
#define HR_BLOCK_SIZE 1024


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

template <typename T, unsigned int ARRAY_SIZE, unsigned int PEAKS_MAX_SIZE>
__global__ void hr_kernel(
    unsigned short int *global_input,
    int *global_input_offsets,
    T *global_output_with_padding,
    T *global_output_hr
)
{   
    int tid = threadIdx.x;
    int index = global_input_offsets[blockIdx.x];
    int input_size = global_input_offsets[blockIdx.x + 1] - global_input_offsets[blockIdx.x];
    int output_with_padding_size = ARRAY_SIZE + SLIDE;

    unsigned short int *input = &global_input[index];    

    __shared__ unsigned short int input_with_padding[ARRAY_SIZE + SLIDE]; // (6'000 + 500) * 2 = 13'000
    int num_elements_to_work_flip = ceil((float) input_size /(float) HR_BLOCK_SIZE);
    flip_and_append_device<unsigned short int>(input, input_with_padding,input_size,  SLIDE, num_elements_to_work_flip);
    // __syncthreads();


    unsigned short int *input_ptr = &input_with_padding[SLIDE];

    T * output_padding_ptr = &global_output_with_padding[get_index_global_array_device(output_with_padding_size, 0)];

    if (tid == 0)
    {
        butter_lowpass_filter_device<unsigned short int, T>(
            input_with_padding, 
            output_padding_ptr,
            input_size + SLIDE
        );

    }
    __syncthreads();
    
    T *output_ptr = &output_padding_ptr[SLIDE];

    __shared__ float energy_array[ARRAY_SIZE]; // 6'000 * 4 = 24'000

    int num_elements_to_work_energy = ceil((float) input_size /(float) HR_BLOCK_SIZE);
    int energy_size = (input_size - 30 + 1);
    int output_size = input_size;

    compute_energy_device<T, float>(
        output_ptr, 
        energy_array,
        output_size,
        energy_size, 
        num_elements_to_work_energy
    );

    __syncthreads();

    int num_elements_to_work_smooth = ceil((float) energy_size /(float) HR_BLOCK_SIZE);

    smooth_device<float, T>(
        energy_array, 
        output_ptr,
        energy_size,
        num_elements_to_work_smooth
    );

    __syncthreads();

    __shared__ unsigned short int peak_indices_array[PEAKS_MAX_SIZE]; // 1'000
    __shared__ int peak_size;
    int output_ptr_size = energy_size;
    int num_elements_to_work_peaks = ceil((float) output_ptr_size /(float) HR_BLOCK_SIZE);

    peak_indices_device<T, unsigned short int>(
        output_ptr, 
        peak_indices_array,
        output_ptr_size,
        &peak_size,
        num_elements_to_work_peaks
    );


    __syncthreads();

    sort_device<unsigned short int>(peak_indices_array, peak_size);

    __syncthreads();

    __shared__ unsigned short int loc[PEAKS_MAX_SIZE];
    __shared__ float pks[PEAKS_MAX_SIZE];

    construct_loc_pks_device<float, unsigned short int>(
        energy_array,
        peak_indices_array,
        loc,
        pks,
        energy_size,
        peak_size
    );

    __syncthreads();

    __shared__ unsigned short int loc_n[PEAKS_MAX_SIZE];
    __shared__ unsigned short int pks_n[PEAKS_MAX_SIZE];

    construct_loc_pksn_device<unsigned short int, unsigned short int>(
        input_ptr,
        loc,
        loc_n,
        pks_n,
        input_size,
        peak_size
    );

    __syncthreads();

    __shared__ int difference[PEAKS_MAX_SIZE];
    __shared__ int zeros_size;

    diff_zeros_count_device<unsigned short int>(
        loc_n,
        difference,
        peak_size,
        &zeros_size
    );

    __syncthreads();

    __shared__ unsigned short int loc_no_dups[PEAKS_MAX_SIZE];
    int no_dups_size = peak_size - zeros_size;

    if (tid == 0)
    {
        __shared__ unsigned short int zeros_ids[PEAKS_MAX_SIZE];
        int j = 0;
        for (int i = 0; i < peak_size - 1; i++)
        {
            if (difference[i] == 0)
            {
                zeros_ids[j] = i;
                j++;
            }
        }

        int k = 0;
        int l = 0;

        for (int i = 0; i < peak_size; i++)
        {
            if (i == zeros_ids[k])
            {
                k++;
            }
            else
            {
                loc_no_dups[l] = loc_n[i];
                l++;
            }
        }
    }
    __syncthreads();

    __shared__ int difference_no_dups[PEAKS_MAX_SIZE];


    diff_device<unsigned short int, int>(
        loc_no_dups,
        difference_no_dups,
        no_dups_size
    );

    __syncthreads();


    int loc_no_dups_size = no_dups_size - 1;

    if (tid == 0 ){
        double sum_hr = 0;
        for (int i = 0; i < loc_no_dups_size; i++)
        {
            double numerator = (100.00 / difference_no_dups[i]) * 60.00;
            sum_hr += numerator;
        }

        double hr = (double) sum_hr / (double) loc_no_dups_size;
        global_output_hr[blockIdx.x] = hr;
    }

}