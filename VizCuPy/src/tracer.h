#pragma once
#include "queue.h"

namespace ftxj {
namespace profiler {

const int deep_stack_threshold = 10;
const int queue_size = 100000;
struct PyTracerLocalResult;
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

  void start(struct timespec* tp);
  void stop();
  PyObject* toPyObj();

  RecordQueue* getQueue();

 private:
  void recordPyCall(PyFrameObject* frame, struct timespec* tp = nullptr);
  void recordPyCCall(PyFrameObject* frame, PyObject* arg);
  void recordPyReturn();
  void recordPyCReturn();
  bool active_{false};
  PyTracerLocalResult* local_results_{nullptr};
  Event* father{nullptr};
  int call_stack_depth{0};
};

// struct CudaTracerLocalResult;
// struct CudaTracer {
//   PyObject_HEAD;

//  public:
//   CudaTracer();
//   ~CudaTracer();

//   void start();
//   void stop();
//   PyObject* toPyObj();
//   void recordCudaCall(const char*);
//   void recordCudaReturn();

//  private:
//   bool activate_{false};
//   Event* father{nullptr};
//   CudaTracerLocalResult* local_results_{nullptr};
//   int call_stack_depth{0};
// };

} // namespace profiler
} // namespace ftxj