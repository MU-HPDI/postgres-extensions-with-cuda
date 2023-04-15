#include "./host_wrapper.hpp"
#include "./host_funcs.cpp"
#include <iostream>
#include <chrono>
#include <thread>

void host_wrapper_heart_rate_estimation(
    std::vector<std::vector<unsigned short int>> selected_filter_vector,
    float *heart_rate,
    int heart_rate_size,
    float *time_elapsed)
{

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < heart_rate_size; i++)
    {

        double result = 0.0;

        unsigned short int *cpu_vec_ptr = &selected_filter_vector[i][0];
        int array_size = selected_filter_vector[i].size();

        cpu_funcs<double, 6'000, 500>(
            cpu_vec_ptr,
            &result,
            array_size);

        heart_rate[i] = (float)result;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    *time_elapsed = elapsed.count() * 1000;
}