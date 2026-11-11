#!/usr/bin/env bash
set -e

echo "=== Initializing CUDA Benchmark Run ==="
mkdir -p data

echo "width,height,block_dim,ms_h2d,ms_kernel,ms_d2h,throughput_gbs,megapixels_sec" > data/metrics.csv

for DIM in 512 1024 2048 4096; do
    for BLOCK in 8 16 32; do
        echo "Executing: Dimension=${DIM}x${DIM}, Block=${BLOCK}x${BLOCK}"
        ./bin/cuda_edge_filter.exe -w ${DIM} -h ${DIM} -b ${BLOCK} -o data/metrics.csv
    done
done

echo "=== Benchmark Suite Finished Successfully ==="
