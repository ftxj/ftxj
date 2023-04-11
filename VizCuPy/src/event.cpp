
#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <string>
#include <vector>
#include "event.h"
namespace ftxj {
namespace profiler {

Event::Event(const EventType& t, PyObject* n, int cat, Event* c)
    : category(cat), type(t), name(n), caller(c) {
  clock_gettime(CLOCK_REALTIME, &tp);
}

std::string getCategoryName(int cat) {
  std::string s = "";
  if (cat & Category::Python) {
    s += "Python";
  }
  if (cat & Category::DeepStack) {
    s += ", DeepStack";
  }
  return s;
}

using time_t = unsigned long long;
double eventTimeStamp(const struct timespec& a, const struct timespec& b) {
  printf("event time 0\n");
  time_t a_sec = static_cast<time_t>(a.tv_sec);
  time_t a_ns = static_cast<time_t>(a.tv_nsec);
  auto a_t = a_sec * 1000 * 1000 * 1000 + a_ns;
  printf("event time 1\n");
  time_t b_base_sec = static_cast<time_t>(b.tv_sec);
  time_t b_base_ns = static_cast<time_t>(b.tv_nsec);
  auto b_t = b_base_sec * 1000 * 1000 * 1000 + b_base_ns;
  printf("event time 2\n");

  return static_cast<double>(a_t - b_t);
}

PyObject* Event::toPyObj(const MetaEvent* meta) {
  PyObject* arg_dict = PyDict_New();
  PyDict_SetItemString(
      arg_dict,
      "ts",
      PyFloat_FromDouble(eventTimeStamp(tp, meta->tp_base) / 1000.0));
  PyDict_SetItemString(arg_dict, "pid", PyLong_FromLong(meta->pid));
  PyDict_SetItemString(arg_dict, "tid", PyLong_FromLong(meta->tid));
  PyDict_SetItemString(arg_dict, "name", name);
  printf("to obj : %s\n", (name));
  PyDict_SetItemString(
      arg_dict, "cat", PyUnicode_FromString(getCategoryName(category).c_str()));
  switch (type) {
    case EventType::PyCall:
    case EventType::PyCCall:
      Py_RETURN_NONE;
      break;
    case EventType::CudaCall:
      PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("B"));
      break;
    case EventType::PyReturn:
    case EventType::PyCReturn:
      if (caller) {
        PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("X"));
        PyDict_SetItemString(
            arg_dict,
            "dur",
            PyFloat_FromDouble(eventTimeStamp(tp, caller->tp) / 1000.0));
      } else {
        printf("bug happen!\n");
        exit(-1);
      }
      break;
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