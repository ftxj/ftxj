#pragma once

#include <Python.h>
#include <iterator>
#include <list>
#include <vector>
namespace ftxj {
namespace profiler {

struct Event;
static const int malloc_size = 100000;
class RecordQueue {
 public:
  RecordQueue(const int&);

  struct Pos {
    int iter{0};
    int bucket{0};
    bool operator<(const Pos& b) {
      if (bucket != b.bucket) {
        return bucket < b.bucket;
      } else {
        return iter < b.iter;
      }
    }
    Pos& operator++() {
      if (iter + 1 < malloc_size) {
        iter++;
      } else {
        iter = 0;
        bucket++;
      }
      return *this;
    }
    Pos& operator--() {
      if (iter - 1 < 0) {
        iter = malloc_size - 1;
        bucket--;
        if (bucket < 0) {
          printf("bug happen in --\n");
          exit(-1);
        }
      } else {
        iter--;
        if (iter < 0) {
          printf("bug happen in --\n");
          exit(-1);
        }
      }
      return *this;
    }
    bool operator>(const Pos& b) {
      if (bucket != b.bucket) {
        return bucket > b.bucket;
      } else {
        return iter > b.iter;
      }
    }
  };

  Event* getNextNeedRecorded();
  Event* getEvent(const Pos&);
  Event* getTopEvent();
  Event* getCallStackTop();
  Pos* getTopEventPos();
  PyObject* toPyObj();

 private:
  int size_;
  int pointer_;
  Event* fast_queue_;
  std::vector<Event*> slow_queue_;
};
} // namespace profiler
} // namespace ftxj