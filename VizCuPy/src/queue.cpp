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

RecordQueue::RecordQueue(const queue_size_t& s) : size_(s), pointer_(0) {
  if (size_ < threshold_size) {
    fast_queue_ = std::vector<Event>(size_);
  } else {
    printf("please shrink the queue size\n");
    exit(-1);
  }
}

Event* RecordQueue::getEvent(const RecordQueue::Pos& pos) {
  if (pos.use_slow) {
    return *std::prev(slow_queue_.begin(), pos.iter);
  } else {
    return &(fast_queue_[pos.iter]);
  }
}

Event* RecordQueue::getTopEvent() {
  if (size_ >= threshold_size) {
    return slow_queue_.front();
  }
  if (pointer_ < 0) {
    printf("bug happen in top\n");
    exit(-1);
  }
  auto r = &fast_queue_[pointer_];
  return r;
}

RecordQueue::Pos* RecordQueue::getTopEventPos() {
  RecordQueue::Pos* pos = new RecordQueue::Pos();
  if (slow_queue_.size() > 0) {
    pos->use_slow = true;
    pos->iter = slow_queue_.size() - 1;
  } else if (pointer_ > 0) {
    pos->use_slow = false;
    pos->iter = pointer_;
  } else {
    printf("bug happen in getTopEventPos\n");
    exit(-1);
  }
  return pos;
}

Event* RecordQueue::getNextNeedRecorded() {
  if (size_ > pointer_) {
    pointer_++;
    return &fast_queue_[pointer_ - 1];
  } else if (size_ <= pointer_ && size_ <= threshold_size) {
    fast_queue_.resize(2 * size_ + 1);
    size_ = size_ * 2 + 1;
    pointer_++;
    return &fast_queue_[pointer_ - 1];
  } else {
    Event* e = new Event();
    slow_queue_.push_back(e);
    return e;
  }
}

PyObject* RecordQueue::toPyObj() {
  PyObject* lst = PyList_New(0);
  for (queue_size_t i = 0; i < pointer_ && i < fast_queue_.size(); ++i) {
    if (auto dic = fast_queue_[i].toPyObj()) {
      PyList_Append(lst, dic);
    }
  }
  for (auto event : slow_queue_) {
    if (auto dic = event->toPyObj()) {
      PyList_Append(lst, dic);
    }
  }
  return lst;
}
} // namespace profiler
} // namespace ftxj