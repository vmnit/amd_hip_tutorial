#include <hip/hip_runtime.h>
//#include <hip/hip_runtime_api.h>
#include <iostream>
#include <cstdlib>

const int BLOCK_SIZE = 16;
const int N = 256;
__global__ void gpu_matrix_multiplication(int *a, int *b, int *c, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int sum = 0;
  if (row < n && col < n) {
    for (int i = 0 ; i < n; i++) {
      sum += a[row*n + i] * b[i*n + col];
    }

    c[row*n + col] = sum;
  }
}

void cpu_matrix_multiplication(int *a, int *b, int *c, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      int sum = 0;
      for (int k = 0; k < n; k++) {
        sum += a[i*n + k] * b[k*n + j];
      }

      c[i*n + j] = sum;
    }
  }
}

int main() {

  int *h_a = new int[N * N];
  int *h_b = new int[N * N];
  int *h_c = new int[N * N];
  int *h_cc = new int[N * N];

  // Initialize Matrix A
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_a[i*N + j] = rand() % RAND_MAX;
    }
  }

  // Initialize Matrix B
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      h_b[i*N + j] = rand() % RAND_MAX;
    }
  }

  int *d_a, *d_b, *d_c;
  hipMalloc((void**) &d_a, sizeof(int) * N * N);
  hipMalloc((void**) &d_b, sizeof(int) * N * N);
  hipMalloc((void**) &d_c, sizeof(int) * N * N);

  hipMemcpy(d_a, h_a, sizeof(int) * N * N, hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, sizeof(int) * N * N, hipMemcpyHostToDevice);

  dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);
  int n_blocks = ceil(N / BLOCK_SIZE);

  dim3 blocksPerGrid (n_blocks, n_blocks);

  gpu_matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  hipDeviceSynchronize();

  hipMemcpy(h_c, d_c, sizeof(int) * N * N, hipMemcpyDeviceToHost);

  cpu_matrix_multiplication(h_a, h_b, h_cc, N);

  bool error = false;
  for (int i = 0; i < N*N; i++) {
    if (h_cc[i] != h_c[i]) {
      std::cout << "Error at index: " << i << std::endl;
      error = true;
      break;
    }
  }

  if (!error) {
    std::cout << "Matrix Multiplication is correct!" << std::endl;
  }

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  delete []h_a;
  delete []h_b;
  delete []h_c;
  delete []h_cc;

  return error;
}
