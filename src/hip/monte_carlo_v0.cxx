#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include "utils/Timer.h"
#include "taskflow/taskflow.hpp"
#include <pthread.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <future>
#include <sstream>

const int N = 100000000;

double monte_carlo_pi_serial(int n) {
#if PROFILE
  Timer t;
#endif
  double x, y;
  double pi = 0;

  std::random_device rd;
  std::mt19937 gen(rd());  // Thread-local generator
  std::uniform_real_distribution<double> dis(-1.0, 1.0);

  for (int i = 0; i < n; i++) {
    x = dis(gen);
    y = dis(gen);

    if (x * x + y * y <= 1) {
      pi++;
    }
  }

#if PROFILE
  std::stringstream ss;
  ss << "monte_carlo_pi_serial for " << n << ": ";

  t.report(ss.str());
#endif

  return 4 * pi / n;
}

__global__ void monte_carlo_pi_gpu(int n, int n_workers, double *final_pi) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_workers) {
    double x, y;
    double pi = 0;

    // Initialize random number generator
    hiprandState state;
    hiprand_init(1234, tid, 0, &state); // Seed, sequence, offset

    for (int i = 0; i < n; i++) {
      x = hiprand_uniform(&state) * 2.0f - 1.0f;  // Random [-1,1]
      y = hiprand_uniform(&state) * 2.0f - 1.0f;  // Random [-1,1]

      if (x * x + y * y <= 1) {
        pi++;
      }
    }

    final_pi[tid] = 4 * pi / n;
  }
}

struct monte_carlo_pi_args {
  int n;
  double pi;
};

void *monte_carlo_pi_thread (void *data) {
  monte_carlo_pi_args *args = (monte_carlo_pi_args *) data;

  args->pi = monte_carlo_pi_serial(args->n);
  return nullptr;
}

double monte_carlo_pi_parallel(int n) {
  int n_threads = std::thread::hardware_concurrency();
  pthread_t threads[n_threads];
  monte_carlo_pi_args args[n_threads];

  for (int t = 0; t < n_threads; t++) {
    args[t].n = n / n_threads;
    pthread_create(&threads[t], nullptr, monte_carlo_pi_thread, &args[t]);
  }

  for (int t = 0; t < n_threads; t++) {
    pthread_join(threads[t], nullptr);
  }

  // Calculate pi
  double pi = 0;
  for (int t = 0; t < n_threads; t++) {
    pi += args[t].pi;
  }
  pi /= n_threads;

  return pi;
}

double monte_carlo_pi_hip(int n) {
  double *d_pi = nullptr;
  int blockSize = 256;
  int gridSize = 256;

  int n_workers = blockSize * gridSize;
  hipMalloc((void **) &d_pi, sizeof(double) * n_workers);

  monte_carlo_pi_gpu<<<gridSize, blockSize>>>(N/n_workers, n_workers, d_pi);

  hipDeviceSynchronize();

  double h_pi[n_workers];

  hipMemcpy(h_pi, d_pi, sizeof(double) * n_workers, hipMemcpyDeviceToHost);

  // Calculate pi
  double pi = 0;
  for (int t = 0; t < n_workers; t++) {
    pi += h_pi[t];
  }
  pi /= n_workers;

  return pi;
}

double monte_carlo_pi_tf(int n) {
  tf::Executor exec;
  tf::Taskflow tf;

  int n_workers = exec.num_workers();
  int n_per_w = n / n_workers;
  std::vector<std::future<double>> vf(n_workers);
  for (int i=0; i < n_workers; i++) {
    vf[i] = exec.async([=]() {
        return monte_carlo_pi_serial(n_per_w);
        });
  }

  exec.wait_for_all();
  // Calculate pi
  double pi = 0;
  for (int t = 0; t < n_workers; t++) {
    pi += vf[t].get();
  }
  pi /= n_workers;

  return pi;
}

int main() {
  Timer t;
  float res_serial = monte_carlo_pi_serial(N);
  t.report("PERF: SERIAL: ");

  t.restart();
  float res_parallel = monte_carlo_pi_parallel(N);
  t.report("PERF: PARALLEL: ");

  t.restart();
  float res_tf = monte_carlo_pi_tf(N);
  t.report("PERF: TASKFLOW: ");

  t.restart();
  float res_hip = monte_carlo_pi_hip(N);
  t.report("PERF: HIP_BASED: ");

  std::cout << "PI serially: " << res_serial << ", parallely: " << res_parallel << ", taskflow: " << res_tf << ", hip: " << res_hip << std::endl;
  return 0;
}
