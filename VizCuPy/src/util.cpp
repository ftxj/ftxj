#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "util.h"

namespace ftxj {
namespace profiler {
namespace util {
// only used in post processing, performace is not important
double eventTimeStamp(const struct timespec& a, const struct timespec& b) {
  time_t a_sec = static_cast<time_t>(a.tv_sec);
  time_t a_ns = static_cast<time_t>(a.tv_nsec);
  auto a_t = a_sec * 1000 * 1000 * 1000 + a_ns;
  time_t b_base_sec = static_cast<time_t>(b.tv_sec);
  time_t b_base_ns = static_cast<time_t>(b.tv_nsec);
  auto b_t = b_base_sec * 1000 * 1000 * 1000 + b_base_ns;

  return static_cast<double>(a_t - b_t);
}

unsigned long long timeDiff(
    const struct timespec& a,
    const struct timespec& b) {
  time_t a_sec = static_cast<time_t>(a.tv_sec);
  time_t a_ns = static_cast<time_t>(a.tv_nsec);
  auto a_t = a_sec * 1000 * 1000 * 1000 + a_ns;

  time_t b_base_sec = static_cast<time_t>(b.tv_sec);
  time_t b_base_ns = static_cast<time_t>(b.tv_nsec);
  auto b_t = b_base_sec * 1000 * 1000 * 1000 + b_base_ns;

  return a_t - b_t;
}

struct timespec maxTime(struct timespec a, struct timespec b) {
  if (a.tv_sec > b.tv_sec) {
    return a;
  } else if (a.tv_sec == b.tv_sec && a.tv_nsec > b.tv_nsec) {
    return a;
  } else {
    return b;
  }
}

struct timespec llu2tp(unsigned long long a) {
  struct timespec b;
  b.tv_nsec = a % (1000 * 1000 * 1000);
  b.tv_sec = a / (1000 * 1000 * 1000);
  return b;
}

unsigned long long tp2llu(struct timespec a) {
  time_t a_sec = static_cast<time_t>(a.tv_sec);
  time_t a_ns = static_cast<time_t>(a.tv_nsec);
  auto a_t = a_sec * 1000 * 1000 * 1000 + a_ns;
  return a_t;
}

} // namespace util
} // namespace profiler
} // namespace ftxj