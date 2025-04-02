#include <hip/hip_runtime.h>
#include <iostream>
#include "utils/Timer.h"

const int N = 100000000;

__global__ void square_kernel(float *input, float *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = input[tid] * input[tid];
}

__global__ void cube_kernel(float *input, float *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = input[tid] * input[tid] * input[tid];
}

int main() {
    Timer t;
    float *inp, *out_sq, *out_cube;
    float *d_inp, *d_out_sq, *d_out_cube;

    inp = new float[N];
    out_sq = new float[N];
    out_cube = new float[N];
    t.report("PERF: after new: ");

    for (int i = 0; i < N; i++) {
        inp[i] = rand() % RAND_MAX;
    }
    t.report("PERF: after rand(): ");

    hipMalloc((void**) &d_inp, sizeof(float) * N);
    hipMalloc((void**) &d_out_sq, sizeof(float) * N);
    hipMalloc((void**) &d_out_cube, sizeof(float) * N);
    t.report("PERF: after hipMalloc(): ");

    hipMemcpy(d_inp, inp, sizeof(float) * N, hipMemcpyHostToDevice);
    t.report("PERF: after hipMemcpy(): ");

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    square_kernel<<<gridSize, blockSize>>>(d_inp, d_out_sq);
    t.report("PERF: after square_kernel(): ");

    cube_kernel<<<gridSize, blockSize>>>(d_inp, d_out_cube);
    t.report("PERF: after cube_kernel(): ");

    hipDeviceSynchronize();
    t.report("PERF: after hipDeviceSynchronize(): ");

    hipMemcpy(out_sq, d_out_sq, sizeof(float) * N, hipMemcpyDeviceToHost);
    t.report("PERF: after hipMemcpy() 1: ");

    hipMemcpy(out_cube, d_out_cube, sizeof(float) * N, hipMemcpyDeviceToHost);
    t.report("PERF: after hipMemcpy() 2: ");

    std::cout << "First element of square: " << out_sq[0] << std::endl;
    std::cout << "First element of cube: " << out_cube[0] << std::endl;

    hipFree(d_inp);
    hipFree(d_out_sq);
    hipFree(d_out_cube);

    delete []inp;
    delete []out_sq;
    delete []out_cube;
    t.report("PERF: after delete(): ");

    return 0;
}
