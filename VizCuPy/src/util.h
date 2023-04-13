#pragma once

#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>

namespace ftxj {
namespace profiler {
namespace util {
double eventTimeStamp(const struct timespec& a, const struct timespec& b);

struct timespec llu2tp(unsigned long long a);

unsigned long long timeDiff(const struct timespec& a, const struct timespec& b);

struct timespec maxTime(struct timespec a, struct timespec b);

unsigned long long tp2llu(struct timespec a);

}; // namespace util

#ifndef NDEBUG
#define ASSERT(condition, message)                                       \
  do {                                                                   \
    if (!(condition)) {                                                  \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__   \
                << " line " << __LINE__ << ": " << message << std::endl; \
      std::terminate();                                                  \
    }                                                                    \
  } while (false)
#else
#define ASSERT(condition, message) \
  do {                             \
  } while (false)
#endif

} // namespace profiler
} // namespace ftxj