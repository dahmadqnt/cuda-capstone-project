#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <getopt.h>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                << " code=" << err << " \"" << cudaGetErrorString(err)   \
                << "\"" << std::endl;                                    \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// 2D Spatial Convolution / Sobel Filter Kernel
__global__ void SobelFilterKernel(const unsigned char* __restrict__ input,
                                  unsigned char* __restrict__ output,
                                  int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
    float gx = (-1.0f * input[(y - 1) * width + (x - 1)]) +
               ( 1.0f * input[(y - 1) * width + (x + 1)]) +
               (-2.0f * input[ y      * width + (x - 1)]) +
               ( 2.0f * input[ y      * width + (x + 1)]) +
               (-1.0f * input[(y + 1) * width + (x - 1)]) +
               ( 1.0f * input[(y + 1) * width + (x + 1)]);

    float gy = (-1.0f * input[(y - 1) * width + (x - 1)]) +
               (-2.0f * input[(y - 1) * width +  x     ]) +
               (-1.0f * input[(y - 1) * width + (x + 1)]) +
               ( 1.0f * input[(y + 1) * width + (x - 1)]) +
               ( 2.0f * input[(y + 1) * width +  x     ]) +
               ( 1.0f * input[(y + 1) * width + (x + 1)]);

    float magnitude = sqrtf(gx * gx + gy * gy);
    magnitude = fminf(fmaxf(magnitude, 0.0f), 255.0f);
    output[y * width + x] = static_cast<unsigned char>(magnitude);
  } else if (x < width && y < height) {
    output[y * width + x] = 0;
  }
}

void PrintUsage(const char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  -w, --width <int>       Image width (default: 1024)\n"
            << "  -h, --height <int>      Image height (default: 1024)\n"
            << "  -b, --block-dim <int>   Thread block dimension (default: 16)\n"
            << "  -o, --output <path>     Output CSV log path (default: data/metrics.csv)\n"
            << "  --help                  Show this help message\n";
}

int main(int argc, char** argv) {
  int width = 1024;
  int height = 1024;
  int block_dim = 16;
  std::string output_csv = "data/metrics.csv";

  static struct option long_options[] = {
      {"width", required_argument, nullptr, 'w'},
      {"height", required_argument, nullptr, 'h'},
      {"block-dim", required_argument, nullptr, 'b'},
      {"output", required_argument, nullptr, 'o'},
      {"help", no_argument, nullptr, '?'},
      {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "w:h:b:o:", long_options, nullptr)) != -1) {
    switch (opt) {
      case 'w': width = std::stoi(optarg); break;
      case 'h': height = std::stoi(optarg); break;
      case 'b': block_dim = std::stoi(optarg); break;
      case 'o': output_csv = optarg; break;
      case '?':
      default:
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }
  }

  size_t num_pixels = static_cast<size_t>(width) * height;
  size_t bytes = num_pixels * sizeof(unsigned char);

  std::cout << "=== CUDA 2D Sobel Filter ===\n";
  std::cout << "Dimensions: " << width << " x " << height << " (" << num_pixels << " pixels)\n";
  std::cout << "Block Dim : " << block_dim << " x " << block_dim << "\n";

  std::vector<unsigned char> h_input(num_pixels);
  std::vector<unsigned char> h_output(num_pixels, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      h_input[y * width + x] = static_cast<unsigned char>(((x / 32) + (y / 32)) % 2 == 0 ? 220 : 30);
    }
  }

  unsigned char *d_input = nullptr, *d_output = nullptr;
  CHECK_CUDA(cudaMalloc(&d_input, bytes));
  CHECK_CUDA(cudaMalloc(&d_output, bytes));

  cudaEvent_t start_h2d, stop_h2d;
  CHECK_CUDA(cudaEventCreate(&start_h2d));
  CHECK_CUDA(cudaEventCreate(&stop_h2d));
  CHECK_CUDA(cudaEventRecord(start_h2d));
  CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaEventRecord(stop_h2d));
  CHECK_CUDA(cudaEventSynchronize(stop_h2d));

  float ms_h2d = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms_h2d, start_h2d, stop_h2d));

  dim3 block(block_dim, block_dim);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  cudaEvent_t start_kernel, stop_kernel;
  CHECK_CUDA(cudaEventCreate(&start_kernel));
  CHECK_CUDA(cudaEventCreate(&stop_kernel));

  CHECK_CUDA(cudaEventRecord(start_kernel));
  SobelFilterKernel<<<grid, block>>>(d_input, d_output, width, height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop_kernel));
  CHECK_CUDA(cudaEventSynchronize(stop_kernel));

  float ms_kernel = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel));

  cudaEvent_t start_d2h, stop_d2h;
  CHECK_CUDA(cudaEventCreate(&start_d2h));
  CHECK_CUDA(cudaEventCreate(&stop_d2h));
  CHECK_CUDA(cudaEventRecord(start_d2h));
  CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaEventRecord(stop_d2h));
  CHECK_CUDA(cudaEventSynchronize(stop_d2h));

  float ms_d2h = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms_d2h, start_d2h, stop_d2h));

  double throughput_gbs = (2.0 * bytes) / (ms_kernel * 1e-3) / 1e9;
  double megapixels_sec = (num_pixels / 1e6) / (ms_kernel * 1e-3);

  std::cout << "Kernel Execution Time: " << ms_kernel << " ms\n";
  std::cout << "Throughput: " << throughput_gbs << " GB/s (" << megapixels_sec << " MPix/s)\n";

  std::ofstream csv(output_csv, std::ios::app);
  if (csv.is_open()) {
    csv << width << "," << height << "," << block_dim << ","
        << ms_h2d << "," << ms_kernel << "," << ms_d2h << ","
        << throughput_gbs << "," << megapixels_sec << "\n";
    csv.close();
    std::cout << "Saved metrics to " << output_csv << "\n";
  }

  CHECK_CUDA(cudaFree(d_input));
  CHECK_CUDA(cudaFree(d_output));
  CHECK_CUDA(cudaEventDestroy(start_h2d));
  CHECK_CUDA(cudaEventDestroy(stop_h2d));
  CHECK_CUDA(cudaEventDestroy(start_kernel));
  CHECK_CUDA(cudaEventDestroy(stop_kernel));
  CHECK_CUDA(cudaEventDestroy(start_d2h));
  CHECK_CUDA(cudaEventDestroy(stop_d2h));

  return EXIT_SUCCESS;
}
