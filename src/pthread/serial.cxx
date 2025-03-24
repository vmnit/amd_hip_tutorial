#include <cstdlib>
#include <iostream>

double monte_carlo_pi_serial (int n) {
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

int main() {
    std::cout << monte_carlo_pi_serial(1000);
    return 0;
}