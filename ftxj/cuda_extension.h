#pragma once

#include <cuda_runtime.h>

namespace ftxj {
namespace cuExt {
class cuTimer {
public:
  cudaEvent_t start;
  cudaEvent_t stop;

  cuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~cuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void Start() { cudaEventRecord(start, 0); }
  void Stop() { cudaEventRecord(stop, 0); }
  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed / 1000;
  }
};
} // namespace cuExt
} // namespace ftxj
