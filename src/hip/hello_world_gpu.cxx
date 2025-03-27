#include <hip/hip_runtime.h>

__global__ void gpuHello() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world from thread %d\n", id);
}

int main() {
    gpuHello<<<1, 1>>>();
    hipDeviceSynchronize();

    return 0;
}