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

  void record(
      const EventType& type,
      PyObject* name,
      int cat,
      Event* other = nullptr);
  PyObject* toPyObj();
  void setMeta();
  Event* top();

 private:
  void recordFast(const EventType& type, PyObject* name, int cat, Event* other);
  void recordSlow(const EventType& type, PyObject* name, int cat, Event* other);

  queue_size_t size_;
  queue_size_t pointer_;
  queue_size_t real_p_;

  const MetaEvent* meta;
  std::vector<Event> fast_queue_;
  std::list<Event*> slow_queue_;
};
} // namespace profiler
} // namespace ftxj