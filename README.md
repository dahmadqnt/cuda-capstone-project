# High-Throughput 2D Spatial Convolution & Sobel Edge Detection in CUDA

## Overview
This repository contains a high-performance 2D spatial convolution and Sobel edge-detection filter implemented in native CUDA C++.
## Project Structure
- src/ : CUDA source code (main.cu) following Google C++ Style Guide standards.
- bin/ : Executable binaries built by Makefile.
- data/ : Benchmark metrics (metrics.csv) containing execution timings and throughput.
- Makefile : Build script with -O3 optimization.
- run.sh : Automated benchmark runner.
- artifacts.tar.gz : Proof of execution archive containing run logs and benchmark metrics.
## Building the Project
Run: make clean && make

## Running the Code
Run CLI directly: ./bin/cuda_edge_filter.exe -w 1024 -h 1024 -b 16 -o data/metrics.csv
Or run the benchmark sweep: ./run.sh
