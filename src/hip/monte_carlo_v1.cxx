#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <iostream>
#include "utils/Timer.hpp"

const int N = 100000000;
const int THREADS_PER_BLOCK = 256;  // Threads per block

// HIP Kernel for Monte Carlo Pi Calculation
__global__ void monte_carlo_pi_kernel(int n, unsigned long long* count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    // Initialize random number generator
    hiprandState state;
    hiprand_init(1234, idx, 0, &state); // Seed, sequence, offset

    unsigned long long local_count = 0;

    for (int i = idx; i < n; i += stride) {
        float x = hiprand_uniform(&state) * 2.0f - 1.0f;  // Random [-1,1]
        float y = hiprand_uniform(&state) * 2.0f - 1.0f;  // Random [-1,1]

        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }

    // Atomic add to global counter
    atomicAdd(count, local_count);
}

// Host function to launch kernel
double monte_carlo_pi_hip(int n) {
    Timer t;
    unsigned long long* d_count;
    unsigned long long h_count = 0;

    t.report("Perf: Before hipMalloc: ");
    hipMalloc(&d_count, sizeof(unsigned long long));
    t.report("Perf: Before hipMemcpy: ");
    hipMemcpy(d_count, &h_count, sizeof(unsigned long long), hipMemcpyHostToDevice);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; // Compute number of blocks

    t.report("Perf: Before hipLaunchKernelGGL: ");
    hipLaunchKernelGGL(monte_carlo_pi_kernel, dim3(blocks), dim3(THREADS_PER_BLOCK), 0, 0, n, d_count);
    t.report("Perf: After hipLaunchKernelGGL: ");
    hipDeviceSynchronize();

    hipMemcpy(&h_count, d_count, sizeof(unsigned long long), hipMemcpyDeviceToHost);
    hipFree(d_count);

    t.report("Perf: End of monte_carlo_pi_hip: ");
    return 4.0 * h_count / n;
}

// Main function
int main() {
    Timer t;
    double pi = monte_carlo_pi_hip(N);
    t.report("PERF: HIP_BASED: ");
    std::cout << "Estimated Pi using HIP (hipRAND): " << pi << std::endl;
    return 0;
}
