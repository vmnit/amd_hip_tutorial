#include<thread>
#include<string>
#include<iostream>

void print_message(int count, std::string msg) {
    for (int i = 0; i < count; i++) {
        std::cout << msg << std::endl;
    }
}

int main() {
    std::thread t1(print_message, 20, "Hello T1");
    std::thread t2(print_message, 15, "Hello T2");

    t1.join();
    t2.join();
    return 0;
}