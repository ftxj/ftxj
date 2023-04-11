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
  real_p_ = 0;
  if (size_ < threshold_size) {
    fast_queue_ = std::vector<Event>(size_);
  }
}

Event* RecordQueue::top() {
  if (size_ >= threshold_size) {
    return slow_queue_.front();
  }
  if (real_p_ < 0) {
    printf("bug happen in top\n");
    exit(-1);
  }
  auto r = &fast_queue_[real_p_];
  real_p_ = real_p_ - 2;
  return r;
}

void RecordQueue::recordFast(
    const EventType& type,
    PyObject* name,
    int cat,
    Event* other) {
  fast_queue_[pointer_].caller = other;
  fast_queue_[pointer_].type = type;
  fast_queue_[pointer_].name = name;
  fast_queue_[pointer_].category = cat;
  clock_gettime(CLOCK_REALTIME, &fast_queue_[pointer_].tp);
  pointer_++;
  real_p_++;
}

void RecordQueue::recordSlow(
    const EventType& type,
    PyObject* name,
    int cat,
    Event* other) {
  Event* e = new Event(type, name, cat, other);
  slow_queue_.push_back(e);
}

void RecordQueue::record(
    const EventType& type,
    PyObject* name,
    int cat,
    Event* other) {
  if (size_ > pointer_) {
    recordFast(type, name, cat, other);
  } else if (size_ <= pointer_ && size_ <= threshold_size) {
    fast_queue_.resize(2 * size_ + 1);
    size_ = size_ * 2 + 1;
    recordFast(type, name, cat, other);
  } else {
    recordSlow(type, name, cat, other);
  }
}

PyObject* RecordQueue::toPyObj() {
  printf("queue to obj\n");
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
  printf("queue to obj end\n");
  return lst;
}
} // namespace profiler
} // namespace ftxj