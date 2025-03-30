#include <hip/hip_runtime.h>
#include <iostream>

const int N = 1000000;
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = new float[N];
    b = new float[N];
    c = new float[N];

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    hipMalloc((void**) &d_a, sizeof(float) * N);
    hipMalloc((void**) &d_b, sizeof(float) * N);
    hipMalloc((void**) &d_c, sizeof(float) * N);

    hipMemcpy(d_a, a, sizeof(float) * N, hipMemcpyHostToDevice);
    hipMemcpy(d_b, b, sizeof(float) * N, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    vector_add<<<gridSize, blockSize>>>(d_c, d_a, d_b, N);

    hipDeviceSynchronize();

    hipMemcpy(c, d_c, sizeof(float) * N, hipMemcpyDeviceToHost);

    std::cout << "First element of result: " << c[0] << std::endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);

    delete []a;
    delete []b;
    delete []c;

    return 0;
}
