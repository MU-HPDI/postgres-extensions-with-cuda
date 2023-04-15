
#ifndef HOST_WRAPPER_H /*/ Include guard */
#define HOST_WRAPPER_H
#include <vector>

/**
 * @brief Finds the heart rate in a short int array.
 * @param selected_filter_vector A vector of vectors of short ints.
 * @param heart_rate A pointer to the output array.
 * @param heart_rate_size The size of the output array.
 * @param time_elapsed A pointer to the time elapsed in (ms)
 */
void host_wrapper_heart_rate_estimation(
    std::vector<std::vector<unsigned short int>> selected_filter_vector,
    float *heart_rate,
    int heart_rate_size,
    float *time_elapsed);

#endif // HOST_WRAPPER_H