#pragma once

namespace ftxj {
namespace profiler {

struct TracerLocalResult;

struct PythonTracer {
  PyObject_HEAD;

 public:
  PythonTracer();
  ~PythonTracer();
  static int profFn(
      PyObject* obj,
      PyFrameObject* frame,
      int what,
      PyObject* arg);

  void start();
  void stop();
  PyObject* toPyObj();

 private:
  void recordPyCall(PyFrameObject* frame);
  void recordPyCCall(PyFrameObject* frame, PyObject* arg);
  void recordPyReturn(PyFrameObject* frame);
  void recordPyCReturn(PyFrameObject* frame, PyObject* arg);
  bool active_{false};
  TracerLocalResult* local_results_{nullptr};
};

} // namespace profiler
} // namespace ftxj