
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
  PyCReturn
};

struct MetaEvent {
  struct timespec tp_base;
  int pid;
  int tid;
};

struct Event {
  EventType type;
  PyObject* name;
  struct timespec tp;
  Event() {}
  Event(const EventType&, PyObject*);
  PyObject* toPyObj(const MetaEvent*);
};

} // namespace profiler
} // namespace ftxj