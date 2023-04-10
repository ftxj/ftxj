#pragma once

#include <Python.h>
#include <frameobject.h>
#include <list>
#include <vector>

namespace ftxj {
namespace profiler {
using queue_size_t = unsigned long long;

class RecordQueue {
 public:
  RecordQueue(const queue_size_t&, const MetaEvent*);

  void record(const EventType&, PyObject*);
  PyObject* toPyObj();
  void setMeta();

 private:
  void recordFast(const EventType&, PyObject*);
  void recordSlow(const EventType&, PyObject*);

  queue_size_t size_;
  queue_size_t pointer_;
  const MetaEvent* meta;
  std::vector<Event> fast_queue_;
  std::list<Event*> slow_queue_;
};
} // namespace profiler
} // namespace ftxj