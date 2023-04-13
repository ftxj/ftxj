
#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <string>
#include <vector>
#include "event.h"
#include "util.h"
namespace ftxj {
namespace profiler {

// only used in post processing, performace is not important
PyObject* Event::getEventTagName() {
  std::string s = "";
  if (tag_ & EventTag::Python) {
    s += "Python,";
  }
  if (tag_ & EventTag::Cuda) {
    s += "Cuda,";
  }
  if (tag_ & EventTag::Torch) {
    s += "Torch,";
  }
  if (tag_ & EventTag::DeepStack) {
    s += "DeepStack,";
  }
  if (tag_ & EventTag::Triton) {
    s += "Triton,";
  }
  return PyUnicode_FromString(s.c_str());
}
PyObject* Event::getEventTypeForPh() {
  std::string s = "";
  switch (type_) {
    case EventType::PyCall:
    case EventType::PyCCall:
    case EventType::CudaCall:
      s += "B";
      break;
    case EventType::PyReturn:
    case EventType::PyCReturn:
    case EventType::CudaReturn:
      s += "E";
      break;
    case EventType::PyComplete:
    case EventType::PyCComplete:
    case EventType::CudaComplete:
      s += "X";
      break;
    default:
      printf("unknow Event type\n");
      exit(-1);
  }
  return PyUnicode_FromString(s.c_str());
}
PyObject* Event::getPyName() {
  std::string name = "";
  if (name_.co_filename_) {
    name += std::string(PyUnicode_AsUTF8(name_.co_filename_));
    name += ".";
  }
  if (name_.f_globals_name_) {
    name += std::string(PyUnicode_AsUTF8(name_.f_globals_name_));
    name += ".";
  }
  if (name_.co_name_py_) {
    name += std::string(PyUnicode_AsUTF8(name_.co_name_py_));
    if (name_.c_func_name_) {
      name += ".";
    }
  }
  if (name_.c_func_name_) {
    if (name_.m_module_) {
      name += std::string(PyUnicode_AsUTF8(name_.m_module_));
      name += ".";
    }
    name += std::string(name_.c_func_name_);
  }
  return PyUnicode_FromString(name.c_str());
}

Event::Event()
    : type_(EventType::None),
      tag_(EventTag::None),
      name_(),
      caller_(nullptr),
      args_(nullptr),
      dur_(0) {}

Event::~Event() {
  Py_XDECREF(name_.co_name_py_);

  Py_XDECREF(name_.co_filename_);
  Py_XDECREF(name_.f_globals_name_);
  Py_XDECREF(name_.m_module_);
  Py_XDECREF(name_.m_self_);
}
using time_t = unsigned long long;

void Event::recordTime() {
  clock_gettime(CLOCK_REALTIME, &tp_);
}

void Event::ahead(struct timespec tp) {
  auto diff = util::timeDiff(tp_, tp);
  printf("diff %lld\n", diff);
  tp_ = util::llu2tp(diff);
}

void Event::delay(struct timespec tp) {
  tp_.tv_sec += tp.tv_sec;
  tp_.tv_nsec += tp.tv_nsec;
}

void Event::recordDuration() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  dur_ = util::timeDiff(tp, tp_);
}

void Event::recordNameFromCode(PyCodeObject* code) {
  name_.co_name_py_ = code->co_name;
  name_.co_filename_ = code->co_filename;

  Py_XINCREF(name_.co_name_py_);
  Py_XINCREF(name_.co_filename_);
}

void Event::recordGlobalName(PyFrameObject* frame) {
  if (frame->f_globals) {
    name_.f_globals_name_ = PyDict_GetItemString(frame->f_globals, "__name__");
    Py_XINCREF(name_.f_globals_name_);
  }
}

void Event::recordNameFromCode(PyCodeObject* code, PyCFunctionObject* fn) {
  name_.co_name_py_ = code->co_name;
  name_.co_filename_ = code->co_filename;

  Py_XINCREF(name_.co_filename_);
  Py_XINCREF(name_.co_name_py_);

  // name_.m_ml_ = fn->m_ml;
  name_.m_self_ = fn->m_self;
  name_.m_module_ = fn->m_module;
  name_.c_func_name_ = fn->m_ml->ml_name;

  // Py_XINCREF(name_.m_ml_);
  Py_XINCREF(name_.m_self_);
  Py_XINCREF(name_.m_module_);
}

PyObject* Event::toPyObj() {
  PyObject* dict = PyDict_New();
  PyDict_SetItemString(
      dict, "ts", PyFloat_FromDouble(util::tp2llu(tp_) / 1000.0));
  PyDict_SetItemString(dict, "pid", PyLong_FromLong(pid));
  PyDict_SetItemString(dict, "tid", PyLong_FromLong(tid));
  PyDict_SetItemString(dict, "name", getPyName());
  PyDict_SetItemString(dict, "cat", getEventTagName());
  PyDict_SetItemString(dict, "ph", getEventTypeForPh());
  if (type_ == EventType::PyComplete || type_ == EventType::PyCComplete ||
      type_ == EventType::CudaComplete) {
    PyDict_SetItemString(
        dict, "dur", PyFloat_FromDouble(double(dur_) / 1000.0));
  }
  return dict;
}

} // namespace profiler
} // namespace ftxj