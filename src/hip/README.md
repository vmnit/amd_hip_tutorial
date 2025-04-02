# Compilation of hip programs

## Compiler
`/opt/rocm/bin/hipcc`

## Header files
`/opt/rocm/include/hip*`

## Header files for taskflow
`third_party/taskflow`

## Header file for utils
`src/utils`

## Compilation command
```
hipcc monte_carlo_v0.cxx -o monte_carlo_v0 -I../ -I../../third_party/taskflow
```

# Disassemble the binaries
`roc-obj --disassemble --target-id gfx942 sq_cube_with_hipstream`

## Details
`roc-obj-ls sq_cube_with_hipstream`

### Output
```
1       host-x86_64-unknown-linux--                                         file://sq_cube_with_hipstream#offset=16384&size=0
1       hipv4-amdgcn-amd-amdhsa--gfx942                                     file://sq_cube_with_hipstream#offset=16384&size=6648
```

### Find gpu chip name and use the above command for disassemble or use extractor
```
roc-obj-extract file://sq_cube_no_hipstream#offset=16384&size=6648
```

It will generate the .co file which can be disassembled using llvm-obj-dump. But couldn't find the utility in the rocm installation, so roc-obj can be used instead as mentioned above.

