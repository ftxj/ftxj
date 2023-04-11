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
    stop(true);
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

void PythonTracer::start(bool from_py) {
  if (active_) {
    return;
  }
  active_ = true;
  meta = new MetaEvent();
  meta->tid = 0;
  local_results_ = new TracerLocalResult(1000000, meta);
  if (from_py) {
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
  }
  PyEval_SetProfile(PythonTracer::profFn, (PyObject*)this);
  clock_gettime(CLOCK_REALTIME, &local_results_->t_s_);
  GlobalContext::update();
}

void PythonTracer::updateMeta(MetaEvent* m) {
  meta->tp_base = m->tp_base;
  meta->pid = m->pid;
}

void PythonTracer::stop(bool from_py) {
  if (active_) {
    PyEval_SetProfile(nullptr, nullptr);
    active_ = false;
    clock_gettime(CLOCK_REALTIME, &local_results_->t_e_);
    if (from_py) {
      auto frame = PyEval_GetFrame();
      static constexpr auto E = EventType::PyReturn;
      local_results_->record_->record(
          E,
          Util::getTraceNameFromFrame(frame, ".stop"),
          Category::Python,
          local_results_->record_->top());
      // std::vector<PyFrameObject*> current_stack;
      // size_t depth = 0; // Make sure we can't infinite loop.
      // while (frame != nullptr && depth <= 128) {
      //   Py_INCREF(frame);
      //   current_stack.push_back(frame);
      //   frame = PyFrame_GetBack(frame);
      //   depth++;
      // }

      // for (auto it = current_stack.rbegin(); it != current_stack.rend();
      // it++) {
      //   recordPyReturn(*it);

      //   Py_DECREF(*it);
      // }
    }
  }
}

PyObject* PythonTracer::toPyObj() {
  return local_results_->record_->toPyObj();
}

void PythonTracer::recordPyCall(PyFrameObject* frame) {
  static constexpr auto E = EventType::PyCall;
  curect_py_depth++;
  int cat = Category::Python;
  if (curect_py_depth > 10) {
    cat |= Category::DeepStack;
  }
  local_results_->record_->record(
      E, Util::getTraceNameFromFrame(frame), cat, nullptr);
}

void PythonTracer::recordPyCCall(PyFrameObject* frame, PyObject* args) {
  static constexpr auto E = EventType::PyCCall;
  local_results_->record_->record(
      E, Util::getTraceNameFromFrame(frame, args), Category::None, nullptr);
}

void PythonTracer::recordPyReturn(PyFrameObject* frame) {
  static constexpr auto E = EventType::PyReturn;
  int cat = Category::Python;
  if (curect_py_depth > 10) {
    cat |= Category::DeepStack;
  }
  curect_py_depth--;
  local_results_->record_->record(
      E,
      Util::getTraceNameFromFrame(frame),
      cat,
      local_results_->record_->top());
}

void PythonTracer::recordPyCReturn(PyFrameObject* frame, PyObject* args) {
  static constexpr auto E = EventType::PyCReturn;
  local_results_->record_->record(
      E,
      Util::getTraceNameFromFrame(frame, args),
      Category::None,
      local_results_->record_->top());
}

} // namespace profiler
} // namespace ftxj