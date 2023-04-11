#pragma once
namespace ftxj {
namespace profiler {

struct TracerLocalResult;
struct MetaEvent;
struct Event;
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

  void start(bool from_py);
  void stop(bool from_py);
  PyObject* toPyObj();

  void updateMeta(MetaEvent*);

 private:
  void recordPyCall(PyFrameObject* frame);
  void recordPyCCall(PyFrameObject* frame, PyObject* arg);
  void recordPyReturn(PyFrameObject* frame);
  void recordPyCReturn(PyFrameObject* frame, PyObject* arg);
  bool active_{false};
  TracerLocalResult* local_results_{nullptr};
  MetaEvent* meta{nullptr};
  int curect_py_depth{0};
};

struct CudaTracerLocalResult;
struct CudaTracer {
  PyObject_HEAD;

 public:
  CudaTracer();
  ~CudaTracer();

  void start(bool from_py);
  void stop(bool from_py);
  PyObject* toPyObj();

  void updateMeta(MetaEvent*);

 private:
  bool activate_{false};
  CudaTracerLocalResult* local_results_{nullptr};
  MetaEvent* meta{nullptr};
};

} // namespace profiler
} // namespace ftxj