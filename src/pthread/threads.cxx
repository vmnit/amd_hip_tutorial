#include <pthread.h>
#include <cstdlib>
#include <iostream>

double monte_carlo_pi_serial(int n) {
    double x, y;
    double pi = 0;

    for (int i = 0; i < n; i++) {
        x = (double) rand() / RAND_MAX * 2 - 1;
        y = (double) rand() / RAND_MAX * 2 - 1;

        if (x * x + y * y <= 1) {
            pi++;
        }
    }

    return 4 * pi / n;
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
    int n_threads = 4;
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

int main() {
    std::cout << monte_carlo_pi_serial(1000);
    return 0;
}