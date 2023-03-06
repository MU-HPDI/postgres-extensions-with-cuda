#include <math.h>
#include "cuda_kernel.hpp"


__global__
void vector_add_cuda(int *x, int *y, int * result, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  result[i] = x[i] + y[i];
}

