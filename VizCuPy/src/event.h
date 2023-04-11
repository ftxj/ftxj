
#pragma once
#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>

namespace ftxj {
namespace profiler {

enum class EventType : uint8_t {
  TorchOp = 0,
  Allocation,
  OutOfMemory,
  PyCall,
  PyCCall,
  PyReturn,
  PyCReturn,
  CudaCall,
  CudaReturn
};

enum Category {
  None = 0,
  Python = 1,
  Cuda = 1 << 1,
  Torch = 1 << 2,
  DeepStack = 1 << 3
}; // namespace Category

struct MetaEvent {
  struct timespec tp_base;
  int pid;
  int tid;
};

struct Event {
  EventType type;
  PyObject* name;
  Event* caller;
  int category;
  struct timespec tp;
  Event() {}
  Event(
      const EventType&,
      PyObject*,
      int cat = Category::None,
      Event* e = nullptr);
  PyObject* toPyObj(const MetaEvent*);
};

} // namespace profiler
} // namespace ftxj