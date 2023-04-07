#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <string>

#include "cpython_version.h"

namespace ftxj {
namespace profiler {

class PyFrameLog {
  struct timespec t_;
};

class Logging {
 public:
  enum LogMode {
    DEBUG,
    INFO,
  };
  static LogMode Debug = DEBUG;
  static LogMode Info = INFO;

  static logging(const std::string&, LogMode&);
};

} // namespace profiler
} // namespace ftxj