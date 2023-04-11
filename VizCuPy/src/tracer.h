#pragma once
namespace ftxj {
namespace profiler {

struct TracerLocalResult;
struct MetaEvent;

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

  void start(MetaEvent*);
  void stop();
  PyObject* toPyObj();

 private:
  void recordPyCall(PyFrameObject* frame);
  void recordPyCCall(PyFrameObject* frame, PyObject* arg);
  void recordPyReturn(PyFrameObject* frame);
  void recordPyCReturn(PyFrameObject* frame, PyObject* arg);
  bool active_{false};
  TracerLocalResult* local_results_{nullptr};
  MetaEvent* meta{nullptr};
};

struct CudaTracerLocalResult;
struct CudaTracer {
  PyObject_HEAD;

 public:
  CudaTracer();
  ~CudaTracer();

  void start(MetaEvent*);
  void stop();
  PyObject* toPyObj();

 private:
  bool activate_{false};
  CudaTracerLocalResult* local_results_{nullptr};
  MetaEvent* meta{nullptr};
};

} // namespace profiler
} // namespace ftxj