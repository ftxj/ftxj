
#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "event.h"

namespace ftxj {
namespace profiler {

Event::Event(const EventType& t, PyObject* n) : type(t), name(n) {
  clock_gettime(CLOCK_REALTIME, &tp);
}

using time_t = unsigned long long;
time_t eventTimeStamp(const struct timespec& a, const struct timespec& b) {
  time_t a_sec = static_cast<time_t>(a.tv_sec);
  time_t a_ns = static_cast<time_t>(a.tv_nsec);
  auto a_t = a_sec * 1000 * 1000 * 1000 + a_ns;
  time_t b_base_sec = static_cast<time_t>(b.tv_sec);
  time_t b_base_ns = static_cast<time_t>(b.tv_nsec);
  auto b_t = b_base_sec * 1000 * 1000 * 1000 + b_base_ns;
  return a_t - b_t;
}

PyObject* Event::toPyObj(const MetaEvent* meta) {
  PyObject* arg_dict = PyDict_New();
  PyDict_SetItemString(
      arg_dict,
      "ts",
      PyFloat_FromDouble(eventTimeStamp(tp, meta->tp_base) / 1000));
  PyDict_SetItemString(arg_dict, "pid", PyLong_FromLong(meta->pid));
  PyDict_SetItemString(arg_dict, "tid", PyLong_FromLong(meta->tid));
  PyDict_SetItemString(arg_dict, "name", name);
  switch (type) {
    case EventType::PyCall:
    case EventType::PyCCall:
    case EventType::CudaCall:
      PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("B"));
      break;
    case EventType::PyReturn:
    case EventType::PyCReturn:
    case EventType::CudaReturn:
      PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("E"));
      break;
    default:
      printf("unknow Event type\n");
      exit(-1);
  }
  return arg_dict;
}

} // namespace profiler
} // namespace ftxj