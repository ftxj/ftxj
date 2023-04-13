
#pragma once
#include <Python.h>
#include <vector>
#include "queue.h"
namespace ftxj {
namespace profiler {

enum class ScheduleType : uint8_t { Split = 0, Align };

struct Event;

class TimeLineSchedule {
 public:
  TimeLineSchedule(RecordQueue* q) : queue(q) {}
  void split();
  void align();
  void update_queue_data(struct timespec tp);
  PyObject* toPyObj();

 private:
  std::vector<std::vector<RecordQueue::Pos*>> getAlignForEachPid();
  RecordQueue* queue;
  std::vector<Event*> split_events_;
  std::vector<RecordQueue::Pos*> split_events_idx_;
  std::vector<RecordQueue::Pos*> align_events_idx_;
  std::vector<Event*> align_events_;
};

} // namespace profiler
} // namespace ftxj