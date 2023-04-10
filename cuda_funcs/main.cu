#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>

#include "cuda_wrappers.hpp"

#define ARRAY_SIZE 512
#define SHORT_MAX 30'000
#define SHORT_MIN 0


int main(int argc, char **argv)
{

    const  int seed = 42;

    // Create a Mersenne Twister random number generator with the given seed
    std::mt19937 gen(seed);

    // Create a uniform distribution of short int values between SHORT_MIN and SHORT_MAX
    std::uniform_int_distribution<short int> dist(SHORT_MIN, SHORT_MAX);

    std::vector<short int> cuda_vec;

    int rows = 10;
    int cols = ARRAY_SIZE;

    for (int i = 0; i < rows * cols; i++)
    {
        cuda_vec.push_back(dist(gen));
    }


    const short int *cuda_vec_ptr = cuda_vec.data();
    short int * max_values = new short int[rows];

    cuda_find_max(cuda_vec_ptr, max_values, rows, cols);


    for (int i = 0; i < rows; i++)
    {
        std::cout << "Max value for row " << i << " is " << max_values[i] << std::endl;
    }

    return 0;
}