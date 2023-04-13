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

struct PyTracerLocalResult {
  RecordQueue* record_{nullptr};
  struct timespec tp_start_;
  struct timespec tp_end_;

  PyTracerLocalResult(const size_t& size) {
    record_ = new RecordQueue(size);
  }

  PyObject* toPyObj() {
    return record_->toPyObj();
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
  if (obj == nullptr) {
    printf("unknow bugs in profFn\n");
    exit(-1);
  }
  auto tracer = reinterpret_cast<PythonTracer*>(obj);
  if (!tracer) {
    printf("unknow bugs in PythonTracer::profFn\n");
    exit(-1);
  }
  switch (what) {
    case PyTrace_CALL:
      // arg always None
      tracer->recordPyCall(frame);
      break;

    case PyTrace_C_CALL:
      // arg are function object being called
      tracer->recordPyCCall(frame, arg);
      break;

    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      // arg are returned value of called function
      tracer->recordPyReturn();
      break;

    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      // arg are function object being called
      tracer->recordPyCReturn();
      break;

    case PyTrace_OPCODE:
    case PyTrace_LINE:
      printf("we doesn't support trace opcode/line\n");
      exit(-1);

    default:
      printf("unknow event when tracing\n");
      exit(-1);
  }
  return 0;
}

void PythonTracer::start(struct timespec* tp) {
  ASSERT(active_ == false, "PythonTracer start twice\n");
  active_ = true;
  local_results_ = new PyTracerLocalResult(queue_size);
  std::vector<PyFrameObject*> current_stack;
  auto frame = PyEval_GetFrame();
  while (frame != nullptr) {
    Py_INCREF(frame);
    current_stack.push_back(frame);
    frame = PyFrame_GetBack(frame);
  }
  for (auto it = current_stack.rbegin(); it != current_stack.rend(); ++it) {
    recordPyCall(*it, tp);
    Py_DECREF(*it);
  }
  PyEval_SetProfile(PythonTracer::profFn, (PyObject*)this);
}
RecordQueue* PythonTracer::getQueue() {
  return local_results_->record_;
}
void PythonTracer::stop() {
  ASSERT(active_, "PythonTracer stop twice\n");
  active_ = false;
  while (call_stack_depth > 0) {
    recordPyReturn();
  }
  PyEval_SetProfile(nullptr, nullptr);
}

PyObject* PythonTracer::toPyObj() {
  PyObject* dict = PyDict_New();
  PyDict_SetItemString(dict, "traceEvents", local_results_->toPyObj());
  PyDict_SetItemString(dict, "displayTimeUnit", PyUnicode_FromString("us"));
  return dict;
}

void PythonTracer::recordPyCall(PyFrameObject* frame, struct timespec* tp) {
  static constexpr auto E = EventType::PyCall;
  EventTag T = EventTag::Python;
  PyCodeObject* code = PyFrame_GetCode(frame);
  if (call_stack_depth > deep_stack_threshold) {
    T = static_cast<EventTag>(EventTag::Python | EventTag::DeepStack);
  }
  Event* event = local_results_->record_->getNextNeedRecorded();
  event->caller_ = father;
  event->type_ = E;
  event->tag_ = T;
  if (tp) {
    event->tp_.tv_sec = tp->tv_sec;
    event->tp_.tv_nsec = tp->tv_nsec;
  } else {
    event->recordTime();
  }
  event->recordNameFromCode(code);
  event->recordGlobalName(frame);
  father = event;
  call_stack_depth++;
  Py_XDECREF(code);
}

void PythonTracer::recordPyCCall(PyFrameObject* frame, PyObject* args) {
  static constexpr auto E = EventType::PyCCall;
  EventTag T = EventTag::C;
  auto code = PyFrame_GetCode(frame);
  auto fn = reinterpret_cast<PyCFunctionObject*>(args);
  if (call_stack_depth > deep_stack_threshold) {
    T = static_cast<EventTag>(EventTag::C | EventTag::DeepStack);
  }
  Event* event = local_results_->record_->getNextNeedRecorded();
  event->caller_ = father;
  event->type_ = E;
  event->tag_ = T;
  event->recordNameFromCode(code, fn);
  event->recordGlobalName(frame);
  event->recordTime();
  father = event;
  call_stack_depth++;
  Py_XDECREF(code);
}

void PythonTracer::recordPyReturn() {
  static constexpr auto E = EventType::PyComplete;
  if (father == nullptr || call_stack_depth <= 0) {
    printf("bug happen\n");
    exit(-1);
  }
  call_stack_depth--;
  father->recordDuration();
  father->type_ = E;
  father = father->caller_;
}

void PythonTracer::recordPyCReturn() {
  static constexpr auto E = EventType::PyCComplete;
  if (father == nullptr || call_stack_depth <= 0) {
    printf("bug happen\n");
    exit(-1);
  }
  call_stack_depth--;
  father->recordDuration();
  father->type_ = E;
  father = father->caller_;
}

} // namespace profiler
} // namespace ftxj