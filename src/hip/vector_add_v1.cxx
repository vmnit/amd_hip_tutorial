#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include "utils/Timer.hpp"

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main() {
    Timer t;

    // Get number of available partitioned devices
    int num_devices = 0;
    hipGetDeviceCount(&num_devices);
    std::cout << "Number of available GPUs: " << num_devices << "\n";

    // Size of vectors
    constexpr int N = 1 << 20;

    // Host vectors
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    std::vector<float> h_C(N, 0.0f);

    // Partition data between GPUs
    int chunk_size = N / num_devices;

    // Store device pointers and GPU streams
    std::vector<float*> A(num_devices);
    std::vector<float*> B(num_devices);
    std::vector<float*> C(num_devices);

    std::vector<hipStream_t> streams(num_devices);

    t.report("PERF: After streams(): ");
    // Launch computations on each GPU partition on separate streams
    for (int dev = 0; dev < num_devices; ++dev) {
        hipSetDevice(dev);  // Set active device

        int offset = dev * chunk_size;
        int size = (dev == num_devices - 1) ? (N - offset) : chunk_size;

        // Allocate device memory
        hipMalloc(&A[dev], size * sizeof(float));
        hipMalloc(&B[dev], size * sizeof(float));
        hipMalloc(&C[dev], size * sizeof(float));
        // t.report("PERF: After hipMalloc(): ");

        // Create stream
        hipStreamCreate(&streams[dev]);

        // Copy data to device
        hipMemcpyAsync(A[dev], h_A.data() + offset, size * sizeof(float), hipMemcpyHostToDevice, streams[dev]);
        hipMemcpyAsync(B[dev], h_B.data() + offset, size * sizeof(float), hipMemcpyHostToDevice, streams[dev]);
        // t.report("PERF: After hipMemcpyAsync(): ");

        // Launch kernel on GPU/GPU partition
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        hipLaunchKernelGGL(vector_add, dim3(grid_size), dim3(block_size), 0, streams[dev], A[dev], B[dev], C[dev], size);
        // t.report("PERF: After hipLaunchKernelGGL(): ");

        // Copy result back to host
        hipMemcpyAsync(h_C.data() + offset, C[dev], size * sizeof(float), hipMemcpyDeviceToHost, streams[dev]);
        // t.report("PERF: After hipMemcpyAsync(): ");
    }
    t.report("PERF: After loop(): ");

    // Synchronize all streams
    for (int dev = 0; dev < num_devices; ++dev) {
        hipSetDevice(dev);
        hipStreamSynchronize(streams[dev]);
    }
    t.report("PERF: After hipStreamSynchronize(): ");

    // Verify results
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != 3.0f) {
            std::cerr << "Verification failed at index: " << i << "\n";
            return -1;
        }
    }
    t.report("PERF: After Verification: ");

    std::cout << "Verification succeeded." << std::endl;

    // Cleanup
    for (int dev = 0; dev < num_devices; ++dev) {
        hipSetDevice(dev);
        hipFree(A[dev]);
        hipFree(B[dev]);
        hipFree(C[dev]);
        hipStreamDestroy(streams[dev]);
    }
    t.report("PERF: After hipStreamDestroy: ");

    return 0;
}
