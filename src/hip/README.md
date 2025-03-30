# Compilation of hip programs

## Compiler
/opt/rocm/bin/hipcc

## Header files
/opt/rocm/include/hip*

## Header files for taskflow
amd_hip_tutorial/third_party/taskflow

## Header file for utils
src/utils

## Compilation command
hipcc monte_carlo_v0.cxx -o monte_carlo_v0 -I../ -I../../third_party/taskflow
