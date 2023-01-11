#pragma once
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

namespace ftxj {
namespace system {
class timer {
private:
  std::chrono::high_resolution_clock::time_point s;
  std::chrono::high_resolution_clock::time_point e;

public:
  void Start() { s = std::chrono::high_resolution_clock::now(); }
  void Stop() { e = std::chrono::high_resolution_clock::now(); }
  auto Elapsed() {
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(e - s);
    return time_span.count();
  }
};
} // namespace system

} // namespace ftxj