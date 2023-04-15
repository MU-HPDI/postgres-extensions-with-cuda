#ifndef CUDA_WRAPPERS_H /* Include guard */
#define CUDA_WRAPPERS_H

#include <vector>

void cuda_wrapper_vector_addition(int *x, int *y, int *result, int n);
/**
 * @brief Finds the maximum values in a short int array.
 *
 * @param data A pointer to the array.
 * @param output A pointer to the output array.
 * @param rows The number of rows in the array.
 * @param cols The number of columns in the array.
 * @return void
 */
void cuda_find_max(const short int *data, short int *output, int rows, int cols);

/**
 * @brief Finds the heart rate in a short int array.
 * @param selected_filter_vector A vector of vectors of short ints.
 * @param heart_rate A pointer to the output array.
 * @param heart_rate_size The size of the output array.
 * @param num_of_threads The number of threads to use.
 * @param time_elapsed A pointer to the time elapsed in (ms)
 */
void cuda_wrapper_heart_rate_estimation(
    std::vector<std::vector<unsigned short int>> selected_filter_vector,
    float *heart_rate,
    int heart_rate_size,
    int num_of_threads,
    float *time_elapsed);

#endif // CUDA_WRAPPERS_H