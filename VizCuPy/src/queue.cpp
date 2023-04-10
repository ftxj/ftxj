#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "queue.h"
#include "tracer.h"

namespace ftxj {
namespace profiler {

static queue_size_t threshold_size =
    queue_size_t(2l * 1024l * 1024l * 1024l) / sizeof(RecordQueue);

RecordQueue::RecordQueue(const queue_size_t& s, const MetaEvent* m)
    : size_(s), pointer_(0), meta(m) {
  if (size_ < threshold_size) {
    fast_queue_ = std::vector<Event>(size_);
  }
}

void RecordQueue::recordFast(const EventType& type, PyObject* name) {
  fast_queue_[pointer_].type = type;
  fast_queue_[pointer_].name = name;
  clock_gettime(CLOCK_REALTIME, &fast_queue_[pointer_].tp);
  pointer_++;
}

void RecordQueue::recordSlow(const EventType& type, PyObject* name) {
  Event* e = new Event(type, name);
  slow_queue_.push_back(e);
}

void RecordQueue::record(const EventType& type, PyObject* name) {
  if (size_ > pointer_) {
    recordFast(type, name);
  } else if (size_ <= pointer_ && size_ <= threshold_size) {
    fast_queue_.resize(2 * size_ + 1);
    size_ = size_ * 2 + 1;
    recordFast(type, name);
  } else {
    recordSlow(type, name);
  }
}

PyObject* RecordQueue::toPyObj() {
  PyObject* lst = PyList_New(0);
  for (queue_size_t i = 0; i < pointer_ && i < fast_queue_.size(); ++i) {
    if (auto dic = fast_queue_[i].toPyObj(meta)) {
      PyList_Append(lst, dic);
    }
  }
  for (auto event : slow_queue_) {
    if (auto dic = event->toPyObj(meta)) {
      PyList_Append(lst, dic);
    }
  }
  return lst;
}
} // namespace profiler
} // namespace ftxj