cmake_minimum_required(VERSION 3.10.0)
project(AMD_HIP_TUTORIAL VERSION 0.1.0 LANGUAGES C CXX)

add_executable(AMD_HIP_TUTORIAL src/pthread/serial.cxx)

include(CTest)
enable_testing()

