#pragma once

#include <Python.h>
#include <iterator>
#include <list>
#include <vector>
namespace ftxj {
namespace profiler {
using queue_size_t = unsigned long long;

struct Event;

class RecordQueue {
 public:
  RecordQueue(const queue_size_t&);

  struct Pos {
    int iter{0};
    bool use_slow{false};
    bool operator<(const Pos& b) {
      if (!use_slow && !b.use_slow) {
        return iter < b.iter;
      } else if (use_slow && b.use_slow) {
        return iter < b.iter;
      } else if (use_slow && !b.use_slow) {
        return false;
      } else {
        return true;
      }
    }
    Pos& operator++() {
      if (use_slow) {
        printf("not implement...\n");
        exit(-1);
      }
      iter++;
      return *this;
    }
    Pos& operator--() {
      if (use_slow) {
        printf("not implement...\n");
        exit(-1);
      }
      iter--;
      return *this;
    }
    bool operator>(int b) {
      return iter > b;
    }
  };

  Event* getNextNeedRecorded();
  Event* getEvent(const Pos&);
  Event* getTopEvent();
  Event* getCallStackTop();
  Pos* getTopEventPos();
  PyObject* toPyObj();

 private:
  queue_size_t size_;
  queue_size_t pointer_;
  std::vector<Event> fast_queue_;
  std::list<Event*> slow_queue_;
};
} // namespace profiler
} // namespace ftxj