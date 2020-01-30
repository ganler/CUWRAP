# CUWRAP: Run Your GPU Engine with Template C++ Frontend

## Quick Start

```shell
mkdir build && cd build
cmake ..
make -j$(nproc) # For Linux
make test
```

## Environment

- C++14 Compiler
- NVCC(CUDA with CXX14)
- CMAKE(3.11+)