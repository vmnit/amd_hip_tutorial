# AMD HIP Tutorial
The repository is based upon the youtube playlist: https://www.youtube.com/playlist?list=PLB1fSi1mbw6IKbZSPz9a2r2DbnHWnLbF-

# HIP programming
For HIP programs, use following command to compile:
```
hipcc <src_filename> -o <out_binary_name>
```
Note: cmake flow is not updated for the above HIP programs.

For more details, please refer `README.md` at location: `src/hip/README.md`

# Using taskflow as submodule for parallel programming, so run following to initialize it
```
git submodule update --init
```
