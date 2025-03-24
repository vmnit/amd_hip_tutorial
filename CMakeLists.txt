cmake_minimum_required(VERSION 3.10.0)
project(AMD_HIP_TUTORIAL VERSION 0.1.0 LANGUAGES C CXX)

add_executable(pthread_serial src/pthread/serial.cxx)

add_executable(pthread_threads src/pthread/threads.cxx)
target_link_libraries(pthread_threads -lpthread)

include(CTest)
enable_testing()

