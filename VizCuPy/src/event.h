
#pragma once
#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>

namespace ftxj {
namespace profiler {

enum class EventType : uint8_t {
  None = 0,
  TorchOp,
  PyCall,
  PyCCall,
  PyReturn,
  PyCReturn,
  CudaCall,
  CudaReturn,
  CudaMem,
  TritonCall,

  PyComplete,
  PyCComplete,
  CudaComplete
};

struct Name {
  PyObject* co_name_py_{nullptr};
  PyObject* co_filename_{nullptr};
  PyObject* f_globals_name_{nullptr};
  PyObject* m_module_{nullptr};
  const char* c_func_name_{nullptr};
  // PyMethodDef* m_ml_{nullptr};
  PyObject* m_self_{nullptr};
};

enum EventTag {
  None = 0,
  Python = 1,
  C = 1 << 2,
  Cuda = 1 << 3,
  Torch = 1 << 4,
  Triton = 1 << 5,
  DeepStack = 1 << 6,
};

struct EventArg {
  const char* code_{nullptr};
};

struct Event {
  Event();
  ~Event();
  void ahead(struct timespec tp);
  void delay(struct timespec tp);
  void recordTime();
  void recordDuration();
  void recordNameFromCode(PyCodeObject*);
  void recordNameFromCode(PyCodeObject*, PyCFunctionObject*);
  void recordGlobalName(PyFrameObject* frame);

  PyObject* getEventTagName();
  PyObject* getEventTypeForPh();
  PyObject* getPyName();

  PyObject* toPyObj();

  EventType type_{EventType::None};
  EventTag tag_{EventTag::None};
  Name name_;
  Event* caller_;
  EventArg* args_;
  int pid{0};
  int tid{0};
  struct timespec tp_;
  unsigned long long dur_;
};

} // namespace profiler
} // namespace ftxj