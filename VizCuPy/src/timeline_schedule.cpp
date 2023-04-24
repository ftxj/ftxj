#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "timeline_schedule.h"
#include "util.h"

namespace ftxj {
namespace profiler {
void TimeLineSchedule::split() {
  Event* event = queue->getTopEvent();
  auto pos = queue->getTopEventPos();
  split_events_.push_back(event);
  split_events_idx_.push_back(pos);
}

void TimeLineSchedule::align() {
  Event* event = queue->getTopEvent();
  auto pos = queue->getTopEventPos();
  align_events_.push_back(event);
  align_events_idx_.push_back(pos);
}

std::vector<std::vector<RecordQueue::Pos*>> TimeLineSchedule::
    getAlignForEachPid() {
  std::vector<std::vector<RecordQueue::Pos*>> res(split_events_idx_.size());
  for (auto idx = 0; idx < align_events_.size(); ++idx) {
    auto e = align_events_[idx];
    auto pos = align_events_idx_[idx];
    res[e->pid].push_back(pos);
  }
  return res;
}

void TimeLineSchedule::update_queue_data(struct timespec tp) {
  printf("begin update...\n");
  RecordQueue::Pos* p = queue->getTopEventPos();
  auto endp = queue->getTopEventPos();
  if (split_events_idx_.size() > 0) {
    p = split_events_idx_[0];
  } else {
    p = endp;
  }
  auto zero = RecordQueue::Pos();
  for (auto j = zero; j < *p; ++j) {
    queue->getEvent(j)->ahead(tp);
  }
  queue->getEvent(*p)->ahead(tp);
  if (split_events_.size() == 0) {
    return;
  }
  for (size_t i = 0; i < split_events_idx_.size(); ++i) {
    RecordQueue::Pos* p = queue->getTopEventPos();
    if (i + 1 < split_events_idx_.size()) {
      p = split_events_idx_[i + 1];
    }
    auto tp = split_events_[i]->tp_;
    for (auto j = *split_events_idx_[i]; j < *p; ++j) {
      queue->getEvent(j)->ahead(tp);
      queue->getEvent(j)->pid = i + 1;
    }
  }

  auto ordered_align = getAlignForEachPid();
  if (ordered_align[0].size() <= 0) {
    return;
  }

  for (size_t j = 0; j < ordered_align[0].size(); ++j) {
    struct timespec max_timer;
    max_timer.tv_sec = 0, max_timer.tv_nsec = 0;
    for (size_t i = 0; i < ordered_align.size(); ++i) {
      auto idx = ordered_align[i][j];
      max_timer = util::maxTime(queue->getEvent(*idx)->tp_, max_timer);
    }
    std::vector<struct timespec> diff_timer;
    for (size_t i = 0; i < ordered_align.size(); ++i) {
      auto idx = ordered_align[i][j];
      auto t =
          util::llu2tp(util::timeDiff(max_timer, queue->getEvent(*idx)->tp_));
      diff_timer.push_back(t);
      unsigned long long a_sec = static_cast<unsigned long long>(t.tv_sec);
      unsigned long long a_ns = static_cast<unsigned long long>(t.tv_nsec);
    }
    auto endp = queue->getTopEventPos();
    for (int k = 0; k < diff_timer.size(); ++k) {
      if (k < split_events_idx_.size()) {
        endp = split_events_idx_[k];
      }
      for (auto bb = *ordered_align[k][j]; bb < *endp; ++bb) {
        queue->getEvent(bb)->delay(diff_timer[k]);
      }
    }
  }
}

} // namespace profiler
} // namespace ftxj