// MIT License
//
// Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "example_utils.hpp"

#include <hip/hip_runtime.h>

#include <iostream>
#include <vector>
#include "utils/Timer.hpp"


#define N 1024 * 256
#define REPEAT 10000000


#include <cmath>

template<unsigned int Width>
__global__ void heavy_trig_kernel(float* out) {
    int idx = (Width + blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = idx * 0.001f;
    for (int i = 0; i < REPEAT; ++i) {
        val = sinf(val) * cosf(val) + tanf(val + 0.0001f);
    }
    out[idx] = val;
}

__global__ void heavy_fib_kernel(float* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float a = 1.0f, b = 1.0f;
    for (int i = 0; i < REPEAT; ++i) {
        float c = a + b + 0.000001f * idx;
        a = b;
        b = c;
    }
    out[idx] = b;
}


__global__ void sqrt_kernel(float* output, const float* input, const int width) {
    // Get the two-dimensional global thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

     if (x < width) {
         if (y < width)
             output[x * width + y] = sqrtf(input[x * width + y]);
     }
}


/// \brief Kernel function to perform matrix transpose using shared memory of constant size
template<unsigned int Width>
__global__ void matrix_transpose_static_shared(float* out, const float* in)
{
    // Shared memory of constant size
    __shared__ float shared_mem[Width * Width];

    // Get the two-dimensional global thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Perform matrix transpose in shared memroy
    shared_mem[y * Width + x] = in[x * Width + y];

    // Synchronize all threads within a thread block
    __syncthreads();

    // Write the matrix transpose into the global memory
    out[y * Width + x] = shared_mem[y * Width + x];
}

/// \brief Kernel function to perform matrix transpose using dynamic shared memory
__global__ void matrix_transpose_dynamic_shared(float* out, const float* in, const int width)
{
    // Dynamic shared memory
    extern __shared__ float shared_mem[];

    // Get the two-dimensional global thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Perform matrix transpose in shared memroy
    shared_mem[y * width + x] = in[x * width + y];

    // Synchronize all threads within a thread block
    __syncthreads();

    // Write the matrix transpose into the global memroy
    out[y * width + x] = shared_mem[y * width + x];
}


int main()
{
    Timer t;

    // Number of streams to be used. It is hardcoded to 2 as this example demonstrates
    // only two kernel launches and their management.
    constexpr unsigned int num_streams = 2;
#if 0
    // Dimension of the input square matrix is width x width
    constexpr unsigned int width = 64;
    constexpr unsigned int size  = width * width;

    // Size in bytes for memory management
    const size_t size_in_bytes = sizeof(float) * size;

    // Allocate host input and output memory as pinned memory using hipHostMalloc.
    // It will ensure that the memory copies will be performed
    // asynchronously when using hipMemcpyAsync

    t.report("PERF: before malloc: ");
    // Allocate host output memory
    std::vector<float*> h_transpose_matrix(num_streams);
    HIP_CHECK(hipHostMalloc(&h_transpose_matrix[0], size_in_bytes));
    HIP_CHECK(hipHostMalloc(&h_transpose_matrix[1], size_in_bytes));

    // Host input memory
    std::vector<float*> h_in(num_streams);
    HIP_CHECK(hipHostMalloc(&h_in[0], size_in_bytes));
    HIP_CHECK(hipHostMalloc(&h_in[1], size_in_bytes));

    t.report("PERF: after malloc: ");
    // Initialize the host input matrix
    for(unsigned int i = 0; i < size; i++)
    {
        h_in[0][i] = static_cast<float>(i);
        h_in[1][i] = static_cast<float>(i);
    }

    // Set the block dimensions
    constexpr unsigned int threads_per_block_x = 4;
    constexpr unsigned int threads_per_block_y = 4;
#endif

   // Create streams
    std::vector<hipStream_t> streams(num_streams);
    for(int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

#if 0
    // Allocate device input and output memory and copy host input data to device memory
    std::vector<float*> d_in(num_streams);
    std::vector<float*> d_transpose_matrix(num_streams);

    t.report("PERF: after initialize matrix: ");
    // Allocate device input memory
    HIP_CHECK(hipMalloc(&d_in[0], size_in_bytes));
    HIP_CHECK(hipMalloc(&d_in[1], size_in_bytes));

    // Allocate device output memory
    HIP_CHECK(hipMalloc(&d_transpose_matrix[0], size_in_bytes));
    HIP_CHECK(hipMalloc(&d_transpose_matrix[1], size_in_bytes));

    t.report("PERF: after hipmalloc: ");
    // hipMemcpyAsync is used without needing to sync before the kernel launch
    // Because both the hipMemcpyAsync and the kernel launch reside in the same stream.
    // The kernel will be executed only after hipMemcpyAsync finishes. There is implicit synchronization.
    // Note: If the host memory is not pinned at allocation time using hipHostMalloc then hipMemcpyAsync
    // will behave as synchronous.
    HIP_CHECK(hipMemcpyAsync(d_in[0], h_in[0], size_in_bytes, hipMemcpyHostToDevice, streams[0]));
    HIP_CHECK(hipMemcpyAsync(d_in[1], h_in[1], size_in_bytes, hipMemcpyHostToDevice, streams[1]));

    // Make sure that Width is evenly divisible by threads_per_block_x and threads_per_block_y
    static_assert(width % threads_per_block_x == 0);
    static_assert(width % threads_per_block_y == 0);

    t.report("PERF: after hipMemcpyAsync: ");
    t.restart();
    for (int ctr=0; ctr < 100000; ctr++) {
      // Launch kernel with stream[0]
      matrix_transpose_static_shared<width>
        <<<dim3(width / threads_per_block_x, width / threads_per_block_y),
        dim3(threads_per_block_x, threads_per_block_y),
        0,
        streams[0]>>>(d_transpose_matrix[0], d_in[0]);

      //t.report("PERF: after matrix_transpose_static_shared on stream 0: ");

      // Launch kernel with stream[1]
      sqrt_kernel<<<dim3(width / threads_per_block_x,
          width / threads_per_block_y),
        dim3(threads_per_block_x, threads_per_block_y),
        sizeof(float) * width * width,
        streams[1]>>>(d_transpose_matrix[1], d_in[1], width);

      //t.report("PERF: after sqrt_kernel on stream 1: ");
    }
    t.report("PERF: after kernel loop: ");

    // Asynchronously copy the results from device to host
    HIP_CHECK(hipMemcpyAsync(h_transpose_matrix[0],
                             d_transpose_matrix[0],
                             size_in_bytes,
                             hipMemcpyDeviceToHost,
                             stream));

    t.report("PERF: after hipMemcpyAsync s0: ");

    HIP_CHECK(hipMemcpyAsync(h_transpose_matrix[1],
                             d_transpose_matrix[1],
                             size_in_bytes,
                             hipMemcpyDeviceToHost,
                             streams[1]));

    t.report("PERF: after hipMemcpyAsync s1: ");

    // Wait for all tasks in both the streams to complete on the device
    HIP_CHECK(hipDeviceSynchronize());
    t.report("PERF: after hipDeviceSynchronize: ");

    std::cout << "After transpose matrix 0 is: " << std::endl;
    //for(unsigned int i = 0; i < width; i++)
    //{
    //  for(unsigned int j = 0; j < width; j++)
    //      std::cout << d_transpose_matrix[0][i*width + j] << " ";
    //  std::cout << std::endl;
    //}
    std::cout << "" << std::endl;

    std::cout << "After transpose matrix 1 is: " << std::endl;
    //for(unsigned int i = 0; i < width; i++)
    //{
    //    for(unsigned int j = 0; j < width; j++)
    //        std::cout << d_transpose_matrix[1][i*width + j] << " ";
    //    std::cout << std::endl;
    //}
    std::cout << std::endl;
    t.report("PERF: after prints: ");

    // Free device memory
    HIP_CHECK(hipFree(d_in[0]));
    HIP_CHECK(hipFree(d_in[1]));
    HIP_CHECK(hipFree(d_transpose_matrix[0]));
    HIP_CHECK(hipFree(d_transpose_matrix[1]));
    t.report("PERF: after hipFree: ");

    // Free host memory
    for(unsigned int i = 0; i < num_streams; i++)
    {
        HIP_CHECK(hipHostFree(h_in[i]));
        HIP_CHECK(hipHostFree(h_transpose_matrix[i]));
    }
    t.report("PERF: after hipHostFree: ");
#endif

    float* h1_out = new float[N];
    float* h2_out = new float[N];
    float* d1_out;
    float* d2_out;
    HIP_CHECK(hipMalloc(&d1_out, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d2_out, N * sizeof(float)));

    //dim3 blockSize(256);
    //dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    dim3 blockSize(256);
    dim3 gridSize(512);

    t.report("PERF: after hipMalloc_2: ");
    // Time measurement
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    t.report("PERF: after hipEventCreate: ");
    t.restart();
    HIP_CHECK(hipEventRecord(start));
    // Launch heavy_fib_kernel
    //heavy_fib_kernel<<<gridSize, blockSize, 0, streams[0]>>>(d1_out);
    heavy_trig_kernel<0><<<gridSize, blockSize, 0, streams[0]>>>(d1_out);
    heavy_trig_kernel<512><<<gridSize, blockSize, 0, streams[1]>>>(d1_out);
    heavy_trig_kernel<0><<<gridSize, blockSize, 0, streams[0]>>>(d2_out);
    heavy_trig_kernel<512><<<gridSize, blockSize, 0, streams[1]>>>(d2_out);
    HIP_CHECK(hipEventRecord(stop));
    t.report("PERF: after hipEventRecord(stop): ");
    
    HIP_CHECK(hipMemcpyAsync(h1_out, d1_out, N * sizeof(float), hipMemcpyDeviceToHost, streams[0]));
    HIP_CHECK(hipMemcpyAsync(h2_out, d2_out, N * sizeof(float), hipMemcpyDeviceToHost, streams[1]));
    t.report("PERF: after hipMemcpyAsync_2: ");

    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    std::cout << "Kernel time: " << ms / 1000.0f << " seconds\n";

    // Check out data
    for (int i=0; i < N; i++) {
      if (h1_out[i] == 0 || h2_out[i] == 0) {
        std::cout << "h1_out: " << h1_out[i] << " and h2_out: " << h2_out[i] << " at i:" << i << std::endl;
        break;
      }
    }

    HIP_CHECK(hipFree(d1_out));
    HIP_CHECK(hipFree(d2_out));
    delete[] h1_out;
    delete[] h2_out;

    // Destroy the streams
    HIP_CHECK(hipStreamDestroy(streams[0]))
    HIP_CHECK(hipStreamDestroy(streams[1]))
    t.report("PERF: after hipStreamDestroy: ");

    std::cout << "streams completed!" << std::endl;
}
