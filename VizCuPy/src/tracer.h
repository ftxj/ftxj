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

//   void start(bool from_py);
//   void stop(bool from_py);
//   PyObject* toPyObj();

//   void updateMeta(MetaEvent*);

//  private:
//   bool activate_{false};
//   CudaTracerLocalResult* local_results_{nullptr};
//   MetaEvent* meta{nullptr};
// };

} // namespace profiler
} // namespace ftxj