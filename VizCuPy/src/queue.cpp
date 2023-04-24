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

static long long threshold_size = 1000000;

RecordQueue::RecordQueue(const int& s) : size_(s), pointer_(-1) {
  if (size_ < threshold_size) {
    fast_queue_ = new Event[size_];
    slow_queue_.push_back(fast_queue_);
  } else {
    printf("OOM happen, please shrink the queue size\n");
    exit(-1);
  }
}

Event* RecordQueue::getEvent(const RecordQueue::Pos& pos) {
  if (pos.bucket >= slow_queue_.size() || pos.iter < 0 || pos.iter >= size_) {
    printf("bug happen on getEvent %d %d\n", pos.bucket, pos.iter);
    exit(-1);
  }
  return &slow_queue_[pos.bucket][pos.iter];
}

Event* RecordQueue::getTopEvent() {
  if (pointer_ < 0 || pointer_ >= size_) {
    printf("bug happen on getTopEvent\n");
    exit(-1);
  }
  return &slow_queue_[slow_queue_.size() - 1][pointer_];
}

RecordQueue::Pos* RecordQueue::getTopEventPos() {
  if (pointer_ < 0 || pointer_ >= size_) {
    printf("bug happen on getTopEventPos\n");
    exit(-1);
  }
  RecordQueue::Pos* pos = new RecordQueue::Pos();
  pos->bucket = slow_queue_.size() - 1;
  pos->iter = pointer_;
  if (pos->iter == 100000) {
    printf("bug on get top\n");
  }
  return pos;
}

Event* RecordQueue::getNextNeedRecorded() {
  if (pointer_ + 1 < size_) {
    pointer_++;
    return &slow_queue_[slow_queue_.size() - 1][pointer_];
  } else {
    printf("resize happen, profiler cost will be high\n");
    fast_queue_ = new Event[size_];
    slow_queue_.push_back(fast_queue_);
    pointer_ = 0;
    return &fast_queue_[pointer_];
  }
}

PyObject* RecordQueue::toPyObj() {
  PyObject* lst = PyList_New(0);
  for (int bucket = 0; bucket < slow_queue_.size() - 1; ++bucket) {
    printf("to obj bucket = %d\n", bucket);
    for (int q = 0; q < size_; ++q) {
      if (auto dic = slow_queue_[bucket][q].toPyObj()) {
        Py_INCREF(dic);
        PyList_Append(lst, dic);
      }
    }
  }
  for (int i = 0; i <= pointer_; ++i) {
    if (auto dic = fast_queue_[i].toPyObj()) {
      Py_INCREF(dic);
      PyList_Append(lst, dic);
    }
  }
  return lst;
}
} // namespace profiler
} // namespace ftxj