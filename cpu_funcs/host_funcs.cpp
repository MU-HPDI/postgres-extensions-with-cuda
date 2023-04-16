
#include <math.h>
#include <stdio.h>
#include <iostream>

#define COEFF_LEN 13
#define SLIDE 500

template <typename T, typename U>
void compute_energy_cpu(T *input, U *energy_array, int input_size, int energy_size, int window_size)
{

    for (int i = 0; i < energy_size; i++)
    {
        double sum = 0;

        for (int j = i; j < i + window_size; j++)
        {
            sum += pow(input[j], 2);
        }

        energy_array[i] = (U)sum;
    }
}

template <typename T, typename U>
void smooth_cpu(const T *y, U *y_smooth, int size_y)
{
    int window_size = 50;
    float factor = 1 / (float)window_size;

    // 1-Dimension convolution operation with padding 'same'
    for (int i = 0; i < size_y; i++)
    {
        if (i < window_size / 2)
        {
            y_smooth[i] = 0;
            for (int j = 0; j < i + window_size / 2 + 1; j++)
            {
                y_smooth[i] += y[j] * factor;
            }
        }
        else if (i >= size_y - window_size / 2)
        {
            y_smooth[i] = 0;
            for (int j = i - window_size / 2; j < size_y; j++)
            {
                y_smooth[i] += y[j] * factor;
            }
        }
        else
        {
            y_smooth[i] = 0;
            for (int j = i - window_size / 2; j < i + window_size / 2 + 1; j++)
            {
                y_smooth[i] += y[j] * factor;
            }
        }
    }
}
template <typename T>
void flip_and_append_cpu(const T *input, T *output, int array_size, int n)
{
    for (int i = 0; i < n; i++)
    {
        output[i] = input[n - i - 1];
    }
    for (int i = 0; i < array_size; i++)
    {
        output[n + i] = input[i];
    }
}

template <typename T, typename U>
void peak_indices_cpu(
    const T *input,
    U *output,
    int input_size,
    int *output_size)
{

    int index = 0;

    for (int i = 1; i < input_size - 1; i++)
    {
        if (input[i] > input[i - 1] && input[i] > input[i + 1])
        {
            output[index] = i;
            index++;
        }
    }

    *output_size = index;
}

template <typename T>
void local_maxima_cpu(
    const T *arr,
    int start,
    int end,
    T *max_value,
    int *max_index)
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
void construct_loc_pks_cpu(
    const T *input_array,
    U *peaks_array,
    U *loc,
    T *pks,
    int input_size,
    int peaks_size)

{

    for (int i = 0; i < peaks_size; i++)
    {
        int peak_value = peaks_array[i];
        T max_value;
        int max_index;

        if (peak_value < 20)
        {
            local_maxima_cpu<T>(input_array, peak_value, peak_value + 30, &max_value, &max_index);
            loc[i] = peak_value + max_index;
        }
        else if (peak_value > input_size - 30)
        {
            local_maxima_cpu<T>(input_array, peak_value - 10, peak_value + 10, &max_value, &max_index);
            loc[i] = peak_value - 21 + max_index;
        }
        else
        {
            local_maxima_cpu<T>(input_array, peak_value - 19, peak_value + 30, &max_value, &max_index);
            loc[i] = peak_value - 20 + max_index;
        }

        pks[i] = max_value;
    }
}

template <typename T, typename U>
void construct_loc_pksn_cpu(
    const T *input_array,
    U *peaks_array,
    U *loc,
    T *pks,
    int input_size,
    int peaks_size)
{

    for (int i = 0; i < peaks_size; i++)
    {
        int peak_value = peaks_array[i];
        T max_value;
        int max_index;

        if (peak_value < 40)
        {
            local_maxima_cpu<T>(input_array, peak_value, peak_value + 30, &max_value, &max_index);
            loc[i] = peak_value + max_index;
        }
        else if (peak_value > input_size - 40)
        {
            local_maxima_cpu<T>(input_array, peak_value - 40, peak_value, &max_value, &max_index);
            loc[i] = peak_value - 41 + max_index;
        }
        else
        {
            local_maxima_cpu<T>(input_array, peak_value - 39, peak_value + 30, &max_value, &max_index);
            loc[i] = peak_value - 40 + max_index;
        }

        pks[i] = max_value;
    }
}

template <typename T, typename U>
T *diff_zeros_count(U *array, int array_size, int *zeros_size)
{

    T *output = new T[array_size - 1];
    int j = 0;
    int zero_counter = 0;

    for (int i = 1; i < array_size; i++, j++)
    {
        output[j] = array[i] - array[i - 1];

        if (output[j] == 0)
        {
            zero_counter++;
        }
    }
    *zeros_size = zero_counter;

    return output;
}

template <typename T>
T *diff_cpu(T *array, int array_size)
{

    T *output = new T[array_size - 1];
    int j = 0;

    for (int i = 1; i < array_size; i++, j++)
    {
        output[j] = array[i] - array[i - 1];
    }

    return output;
}

template <typename Input, typename T>
void butter_lowpass_filter_cpu(
    Input *x,
    T *filter_x,
    int size_x)
{

    T zi[COEFF_LEN];

    T coeff_b[] = {0.000234487894504741, 0, -0.00140692736702845, 0, 0.00351731841757112, 0, -0.00468975789009482, 0, 0.00351731841757112, 0, -0.00140692736702845, 0, 0.000234487894504741};
    T coeff_a[] = {1, -9.60972210409256, 42.5044725167320, -114.495656304045, 209.312140612173, -273.691848572724, 262.545133243944, -186.198726793693, 96.9036651938758, -36.0926373232641, 9.13215305571148, -1.40928057198485, 0.100307047533755};

    for (int i = 0; i < COEFF_LEN; i++)
    {
        zi[i] = (T)0.0;
    }

    for (int m = 0; m < size_x; m++)
    {
        filter_x[m] = coeff_b[0] * x[m] + zi[0];

        for (int i = 1; i < COEFF_LEN; i++)
        {
            zi[i - 1] = coeff_b[i] * x[m] + zi[i] - coeff_a[i] * filter_x[m];
        }
    }
}

template <typename T, unsigned int ARRAY_SIZE, unsigned int PEAKS_MAX_SIZE>
void cpu_funcs(
    unsigned short int *input,
    T *output,
    int input_size)
{

    unsigned short int input_with_padding[ARRAY_SIZE + SLIDE];
    flip_and_append_cpu<unsigned short int>(input, input_with_padding, input_size, SLIDE);

    unsigned short int *input_ptr = &input_with_padding[SLIDE];

    T output_with_padding[ARRAY_SIZE + SLIDE];
    butter_lowpass_filter_cpu<unsigned short int, T>(input_with_padding, output_with_padding, ARRAY_SIZE + SLIDE);

    T *output_ptr = &output_with_padding[SLIDE];
    float energy[ARRAY_SIZE];
    int energy_window_size = 30;
    int energy_size = (input_size - energy_window_size + 1);

    compute_energy_cpu<T, float>(output_ptr, energy, input_size, energy_size, energy_window_size);

    T output_smooth[ARRAY_SIZE];

    smooth_cpu<float, T>(energy, output_smooth, energy_size);

    unsigned short int peak_indices[PEAKS_MAX_SIZE];
    int peak_size;
    peak_indices_cpu<T, unsigned short int>(output_smooth, peak_indices, energy_size, &peak_size);

    unsigned short int loc[PEAKS_MAX_SIZE];
    float pks[PEAKS_MAX_SIZE];
    construct_loc_pks_cpu<float, unsigned short int>(energy, peak_indices, loc, pks, energy_size, peak_size);

    unsigned short int loc_n[PEAKS_MAX_SIZE];
    unsigned short int pks_n[PEAKS_MAX_SIZE];

    construct_loc_pksn_cpu<unsigned short int, unsigned short int>(
        input_ptr,
        loc,
        loc_n,
        pks_n,
        input_size,
        peak_size);

    int zeros_size;
    int *difference = diff_zeros_count<int, unsigned short int>(loc_n, peak_size, &zeros_size);

    int no_dups_size = peak_size - zeros_size;
    int loc_no_dups[no_dups_size];

    int zeros_ids[zeros_size];
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

    int *difference_no_dups = diff_cpu(loc_no_dups, no_dups_size);
    int difference_no_dups_size = no_dups_size - 1;

    double sum_hr = 0;
    for (int i = 0; i < difference_no_dups_size; i++)
    {
        double numerator = 100.00 / difference_no_dups[i];
        sum_hr += numerator * 60.00;
    }

    double avg = (double)sum_hr / (double)(difference_no_dups_size);

    output[0] = avg;
}
