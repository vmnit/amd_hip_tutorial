#include <taskflow/taskflow.hpp>
#include <iostream>
#include <future>
#include <vector>

int my_function() {
  static int i = 0;
  return 42 + i++;
}

int main() {
  tf::Executor executor;
  tf::Taskflow taskflow;

  std::vector<std::future<int>> vf(10);
  for (int i = 0; i < 10; i++)
    vf[i] = executor.async(my_function);

  // Do other work while the task is running

  executor.wait_for_all();

  for (auto& f: vf)
    std::cout << "Result: " << f.get() << std::endl;

  return 0;
}
