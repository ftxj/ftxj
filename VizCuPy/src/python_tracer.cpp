#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "queue.h"
#include "tracer.h"
#include "util.h"

namespace ftxj {
namespace profiler {

namespace GlobalContext {

static int tracing_id = 0;
static void update() {
  tracing_id++;
}

} // namespace GlobalContext

struct TracerLocalResult {
  RecordQueue* record_{nullptr};
  struct timespec t_s_;
  struct timespec t_e_;
  TracerLocalResult(const size_t& size, const MetaEvent* m) {
    record_ = new RecordQueue(size, m);
  }
};

PythonTracer::PythonTracer() {}

PythonTracer::~PythonTracer() {
  if (active_) {
    stop();
  }
}

int PythonTracer::profFn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  auto tracer = reinterpret_cast<PythonTracer*>(obj);
  if (!tracer) {
    printf("unknow bugs \n");
    exit(-1);
  }
  switch (what) {
    case PyTrace_CALL:
      tracer->recordPyCall(frame);
      break;

    case PyTrace_C_CALL:
      tracer->recordPyCCall(frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      tracer->recordPyReturn(frame);
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      tracer->recordPyCReturn(frame, arg);
      break;

    default:
      printf("unknow event when tracing\n");
      exit(-1);
  }
  return 0;
}

void PythonTracer::start() {
  if (active_) {
    return;
  }
  active_ = true;
  MetaEvent* meta = new MetaEvent();
  clock_gettime(CLOCK_REALTIME, &meta->tp_base);
  meta->pid = GlobalContext::tracing_id;
  meta->tid = 0;
  local_results_ = new TracerLocalResult(100000, meta);

  std::vector<PyFrameObject*> current_stack;
  auto frame = PyEval_GetFrame();
  size_t depth = 0; // Make sure we can't infinite loop.
  while (frame != nullptr && depth <= 128) {
    Py_INCREF(frame);
    current_stack.push_back(frame);
    frame = PyFrame_GetBack(frame);
    depth++;
  }

  for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
    recordPyCall(*it);

    Py_DECREF(*it);
  }
  PyEval_SetProfile(PythonTracer::profFn, (PyObject*)this);
  clock_gettime(CLOCK_REALTIME, &local_results_->t_s_);
  GlobalContext::update();
}

void PythonTracer::stop() {
  if (active_) {
    PyEval_SetProfile(nullptr, nullptr);
    active_ = false;
    clock_gettime(CLOCK_REALTIME, &local_results_->t_e_);
    auto frame = PyEval_GetFrame();
    static constexpr auto E = EventType::PyReturn;
    local_results_->record_->record(
        E, Util::getTraceNameFromFrame(frame, ".stop"));

    std::vector<PyFrameObject*> current_stack;
    size_t depth = 0; // Make sure we can't infinite loop.
    while (frame != nullptr && depth <= 128) {
      Py_INCREF(frame);
      current_stack.push_back(frame);
      frame = PyFrame_GetBack(frame);
      depth++;
    }

    for (auto it = current_stack.rbegin(); it != current_stack.rend(); it++) {
      recordPyReturn(*it);

      Py_DECREF(*it);
    }
  }
}

PyObject* PythonTracer::toPyObj() {
  return local_results_->record_->toPyObj();
}

void PythonTracer::recordPyCall(PyFrameObject* frame) {
  static constexpr auto E = EventType::PyCall;
  local_results_->record_->record(E, Util::getTraceNameFromFrame(frame));
}

void PythonTracer::recordPyCCall(PyFrameObject* frame, PyObject* args) {
  static constexpr auto E = EventType::PyCCall;
  local_results_->record_->record(E, Util::getTraceNameFromFrame(frame, args));
}

void PythonTracer::recordPyReturn(PyFrameObject* frame) {
  static constexpr auto E = EventType::PyReturn;
  local_results_->record_->record(E, Util::getTraceNameFromFrame(frame));
}

void PythonTracer::recordPyCReturn(PyFrameObject* frame, PyObject* args) {
  static constexpr auto E = EventType::PyCReturn;
  local_results_->record_->record(E, Util::getTraceNameFromFrame(frame, args));
}

} // namespace profiler
} // namespace ftxj