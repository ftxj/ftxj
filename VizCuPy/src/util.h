#pragma once

#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>

namespace ftxj {
namespace profiler {

class Util {
 public:
  static PyObject* getTraceNameFromFrame(PyFrameObject* frame) {
    PyCodeObject* code = PyFrame_GetCode(frame);
    PyObject* last_name = nullptr;
    if (code) {
      last_name = code->co_name;
    } else {
      last_name = PyUnicode_FromString("<noname>");
    }
    Py_DECREF(code);
    return last_name;
  }

  static PyObject* getTraceNameFromFrame(
      PyFrameObject* frame,
      const char* str) {
    PyCodeObject* code = PyFrame_GetCode(frame);
    PyObject* last_name = nullptr;
    if (code) {
      last_name = PyUnicode_Concat(code->co_name, PyUnicode_FromString(str));
    } else {
      last_name = PyUnicode_FromString("<noname>");
    }
    Py_DECREF(code);
    return last_name;
  }

  static PyObject* getTraceNameFromFrame(PyFrameObject* frame, PyObject* arg) {
    PyCodeObject* code = PyFrame_GetCode(frame);
    PyObject* last_name = nullptr;
    if (code) {
      last_name = code->co_name;
    } else {
      last_name = PyUnicode_FromString("<noname>");
    }
    PyCFunctionObject* cfunc = reinterpret_cast<PyCFunctionObject*>(arg);
    PyObject* cfunc_name = nullptr;
    if (cfunc->m_module) {
      cfunc_name = PyUnicode_FromObject(cfunc->m_module);
      last_name = PyUnicode_Concat(last_name, PyUnicode_FromString("."));
      last_name = PyUnicode_Concat(last_name, cfunc_name);
    }
    if (cfunc->m_ml) {
      cfunc_name = PyUnicode_FromString(cfunc->m_ml->ml_name);
      last_name = PyUnicode_Concat(last_name, PyUnicode_FromString("."));
      last_name = PyUnicode_Concat(last_name, cfunc_name);
    }
    Py_DECREF(code);
    return last_name;
  }
};

} // namespace profiler
} // namespace ftxj