cmake_minimum_required(VERSION 3.10.0)
project(AMD_HIP_TUTORIAL VERSION 0.1.0 LANGUAGES C CXX)

add_executable(pthread_serial src/pthread/serial.cxx)

add_executable(pthread_threads src/pthread/threads.cxx src/utils/Timer.h third_party/taskflow/taskflow/taskflow.hpp)
target_include_directories(pthread_threads PRIVATE src third_party/taskflow)
target_link_libraries(pthread_threads -lpthread)

add_executable(cxx_threads src/cxx_thread/threads.cxx)
target_link_libraries(cxx_threads -pthread)

# set(HIP_INCLUDE_DIR /opt/rocm/include)
# set(HIP_INCLUDE_HEADERS ${HIP_INCLUDE_DIR}/hip/hip_runtime.h)
# add_executable(hip_hello src/hip/hello.cxx ${HIP_INCLUDE_HEADERS})
# target_include_directories(hip_hello PUBLIC ${HIP_INCLUDE_DIR})

include(CTest)
enable_testing()

