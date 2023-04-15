#include <iostream>
#include <math.h>

#define COEFF_LEN 13


__device__ 
int get_index_global_array_device(int arr_size, int offset)
{
    return blockIdx.x * arr_size + offset;
}

template <typename T>
__device__ 
void flip_and_append_device(
    const T* input, 
    T* output, 
    int array_size, 
    int n, 
    int num_elements_to_work
){
    int tid = threadIdx.x;

    if (tid < n)
    {
        output[tid] = input[n - tid - 1];
    }

    __syncthreads();

    int start = tid * num_elements_to_work;
    int end = start + num_elements_to_work > array_size ? array_size : start + num_elements_to_work;

    for (int i = start; i < end; i++)
    {
        output[i + n] = input[i];
    }

}

template <typename Input, typename T>
__device__ 
void butter_lowpass_filter_device(
    const Input *x,
    T *filter_x,
    int size_x
)
{

    int tid = threadIdx.x;

    if (tid != 0)
    {
        return;
    }

    T zi[COEFF_LEN] = {0.0};

    T coeff_b[] = {0.000234487894504741, 0, -0.00140692736702845, 0, 0.00351731841757112, 0, -0.00468975789009482, 0, 0.00351731841757112, 0, -0.00140692736702845, 0, 0.000234487894504741};
    T coeff_a[] = {1, -9.60972210409256, 42.5044725167320, -114.495656304045, 209.312140612173, -273.691848572724, 262.545133243944, -186.198726793693, 96.9036651938758, -36.0926373232641, 9.13215305571148, -1.40928057198485, 0.100307047533755};

    for (int m = 0; m < size_x; m++)
    {
        T x_value = (T)x[m];
        filter_x[m] = coeff_b[0] * x_value + zi[0];

        for (int i = 1; i < COEFF_LEN; i++)
        {
            zi[i - 1] = coeff_b[i] * x_value + zi[i] - coeff_a[i] * filter_x[m];
        }

    }
}

template<typename T, typename U>
__device__ 
void compute_energy_device(
    const T* input, 
    U* energy_array, 
    int input_size,
    int energy_size,
    int num_elements_to_work
)
{
    int tdix = threadIdx.x;
    int start = tdix * num_elements_to_work;
    int end = start + num_elements_to_work > energy_size ? energy_size : start + num_elements_to_work;

    for (int i = start; i < end; i++)
    {
        double sum = 0;

        #pragma unroll
        for (int j = 0; j < 30; j++)
        {
            double value = (double) input[i + j];
            sum += (value * value);
        }

        energy_array[i] = (U)sum;
    }
}

template<typename T, typename U>
__device__ 
void smooth_device(const T* y, U* y_smooth, int size_y, int num_elements_work)
{

    int window_size = 50;
    float factor = 1 /(T)window_size;

    int tdix = threadIdx.x;

    int start = num_elements_work * tdix;
    int end = start + num_elements_work > size_y ? size_y : start + num_elements_work;

    int inner_start, inner_end;
    for (int i = start; i < end; i++)
    {
        y_smooth[i] = 0;
        double sum = 0;
        inner_start = i - window_size / 2;
        inner_end = i + window_size / 2 + 1;

        if (i < window_size / 2)
        {
            inner_start = 0;
        }
        else if (i >= size_y - window_size / 2)
        {
            inner_end = size_y;
        }

        for (int j = inner_start; j < inner_end; j++)
        {
            sum += y[j] * factor;
        }

        y_smooth[i] = (U)sum;
    }

}

template<typename T, typename U>
__device__ 
void peak_indices_device(const T *input, U *output, int input_size, int *output_size, int num_elements_work)
{
    int tdix = threadIdx.x;

    int start = tdix == 0 ? 1 : num_elements_work * tdix;
    int end = start + num_elements_work > input_size ? input_size : start + num_elements_work;

    
    atomicExch(output_size, 0);
    __syncthreads();
    
    for (int i = start; i < end; i++)
    {
        if (input[i] > input[i - 1] && input[i] > input[i + 1])
        {
            int index = atomicAdd(output_size, 1);
            output[index] = i;
        }
    }
    __syncthreads();

}

template<typename T>
__device__ 
void sort_device(T* input, int size)
{
    int tdix = threadIdx.x;

    if (tdix != 0)
    {
        return;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            if (input[i] > input[j])
            {
                T temp = input[i];
                input[i] = input[j];
                input[j] = temp;
            }
        }
    }
}


template <typename T>
__device__ 
void local_maxima(
    const T *arr, 
    int start,
    int end, 
    T *max_value, 
    int *max_index
)
{
    T tmp_max_value = arr[start];
    int tmp_max_index = start;

    for (int j = start; j < end; j++)
    {
        if (arr[j] > tmp_max_value)
        {
            tmp_max_value = arr[j];
            tmp_max_index = j;
        }
    }

    *max_value = tmp_max_value;
    *max_index = tmp_max_index - start;
}

template <typename T, typename U>
__device__ 
void construct_loc_pks_device(
    const T* input_array,
    U *peaks_array,
    U *loc,
    T *pks,
    int input_size,
    int peaks_size
)
{

    int tid = threadIdx.x;
    if (tid >= peaks_size)
    {
        return;
    }

    int peak_value = peaks_array[tid];
    T max_value;
    int max_index;


    if (peak_value < 20)
    {
        local_maxima<T>(input_array, peak_value, peak_value + 30, &max_value, &max_index);
        loc[tid] = peak_value + max_index;
    }
    else if (peak_value > input_size - 30)
    {
        local_maxima<T>(input_array, peak_value - 10, peak_value + 10, &max_value, &max_index);
        loc[tid] = peak_value - 21 + max_index;
    }
    else
    {
        local_maxima<T>(input_array, peak_value - 19, peak_value + 30, &max_value, &max_index);
        loc[tid] = peak_value - 20 + max_index;
    }

    pks[tid] = max_value;
}


template <typename T, typename U>
__device__ 
void construct_loc_pksn_device(
    const T* input_array,
    U *peaks_array,
    U *loc,
    T *pks,
    int input_size,
    int peaks_size
)
{

    int tid = threadIdx.x;
    if (tid >= peaks_size)
    {
        return;
    }

    int peak_value = peaks_array[tid];
    T max_value;
    int max_index;

    if (peak_value < 40)
    {
        local_maxima<T>(input_array, peak_value, peak_value + 30, &max_value, &max_index);
        loc[tid] = peak_value + max_index;
    }
    else if (peak_value > input_size - 40)
    {
        local_maxima<T>(input_array, peak_value - 40, peak_value, &max_value, &max_index);
        loc[tid] = peak_value - 41 + max_index;
    }
    else
    {
        local_maxima<T>(input_array, peak_value - 39, peak_value + 30, &max_value, &max_index);
        loc[tid] = peak_value - 40 + max_index;
    }

    pks[tid] = max_value;
    
}


template <typename T>
__device__ 
void diff_zeros_count_device(
    T *array,
    int *output,
    int array_size,
    int *zero_counter
)
{

    int tid = threadIdx.x;

    atomicExch(zero_counter, 0);
    __syncthreads();

    if (tid >= array_size || tid == 0)
    {
        return;
    }

    output[tid - 1] = array[tid] - array[tid - 1];

    if (array[tid] - array[tid - 1] == 0)
    {
        atomicAdd(zero_counter, 1);
    }

}

template <typename T, typename U>
__device__ 
void diff_device(
    T *array,
    U *output,
    int array_size)
{

    int tid = threadIdx.x;

    if (tid >= array_size)
    {
        return;
    }

    int i = tid + 1;
    output[tid] = array[i] - array[i - 1];

}