#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include "cpython_version.h"
namespace ftxj {
namespace profiler {

class TracerContext;
class PythonTracer final {
 public:
  PythonTracer();
  static int prifile_fn(
      PyObject* obj,
      PyFrameObject* frame,
      int what,
      PyObject* arg);
  void start(TracerContext* ctx);
  void stop(TracerContext* ctx);
  void record_call(PyFrameObject*, TracerContext*, int type);
  void record_return(PyFrameObject*, TracerContext*, int type);

 private:
  bool active_;
};

class Timer {
 public:
  static unsigned long long getTime(bool allow_monotonic = false) {
    struct timespec t {};
    auto mode = CLOCK_REALTIME;
    if (allow_monotonic) {
      mode = CLOCK_MONOTONIC;
    }
    clock_gettime(mode, &t);
    return static_cast<unsigned long long>(t.tv_sec) * 1000000000 +
        static_cast<unsigned long long>(t.tv_nsec);
  }
};

PythonTracer::PythonTracer() {
  active_ = false;
}

struct TracerContext {
  PyObject_HEAD;
  PythonTracer* tracer;
  int call_times;
  unsigned long long start_time;
};

static PyObject* tracer_context_start(TracerContext* self, PyObject* args) {
  self->tracer->start(self);
  Py_RETURN_NONE;
}

static PyObject* tracer_context_stop(TracerContext* self, PyObject* args) {
  self->tracer->stop(self);
  Py_RETURN_NONE;
}

static PyMethodDef Example_methods[] = {
    {"start",
     (PyCFunction)tracer_context_start,
     METH_VARARGS,
     "tracer_context_start function"},
    {"stop",
     (PyCFunction)tracer_context_stop,
     METH_VARARGS,
     "tracer_context_stop function"},
    {NULL, NULL, 0, NULL}};

static PyObject* Example_New(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  printf("new example object\n");
  TracerContext* self = (TracerContext*)type->tp_alloc(type, 0);
  if (self) {
    self->tracer = new PythonTracer;
  }
  return (PyObject*)self;
}

static void Example_delete(TracerContext* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject TracerContextType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ftxj.profiler",
    .tp_basicsize = sizeof(TracerContext),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Example_delete,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Profiler",
    .tp_methods = Example_methods,
    .tp_new = Example_New};

static PyMethodDef example_methods[] = {{NULL, NULL, 0, NULL}};

static struct PyModuleDef example_module =
    {PyModuleDef_HEAD_INIT, "ftxj.profiler", NULL, -1, example_methods};

PyMODINIT_FUNC PyInit_profiler(void) {
  printf("init py example\n");
  if (PyType_Ready(&TracerContextType) < 0) {
    return NULL;
  }
  PyObject* m = PyModule_Create(&example_module);
  if (!m) {
    return NULL;
  }
  Py_INCREF(&TracerContextType);
  if (PyModule_AddObject(m, "Tracer", (PyObject*)&TracerContextType) < 0) {
    Py_DECREF(&TracerContextType);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}

void PythonTracer::start(TracerContext* ctx) {
  active_ = true;
  unsigned long long s = Timer::getTime() / 1000;
  ctx->start_time = s;
  PyEval_SetProfile(PythonTracer::prifile_fn, (PyObject*)ctx);
}

void PythonTracer::stop(TracerContext* ctx) {
  if (active_) {
    PyEval_SetProfile(nullptr, (PyObject*)ctx);
    active_ = false;
  }
}

int PythonTracer::prifile_fn(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  auto ctx = reinterpret_cast<TracerContext*>(obj);
  auto tracer = reinterpret_cast<TracerContext*>(obj)->tracer;
  switch (what) {
    case PyTrace_CALL:
      tracer->record_call(frame, ctx, 0);
      break;
    case PyTrace_C_CALL:
      tracer->record_call(frame, ctx, 1);
      break;
    case PyTrace_EXCEPTION:
    case PyTrace_RETURN:
      tracer->record_return(frame, ctx, 0);
      break;
    case PyTrace_C_EXCEPTION:
    case PyTrace_C_RETURN:
      tracer->record_return(frame, ctx, 1);
      break;
  }
  return 0;
}

void PythonTracer::record_call(
    PyFrameObject* frame,
    TracerContext* ctx,
    int type) {
  PyCodeObject* code = PyFrame_GetCode(frame);
  PyObject* co_name = code->co_name;
  const char* name = PyUnicode_AsUTF8(co_name);
  ctx->call_times += 1;
  unsigned long long s = Timer::getTime() / 1000;
  auto gap = s - ctx->start_time;
  printf(
      "stack = %d, mode = %d, call %s, time = %llu\n",
      ctx->call_times,
      type,
      name,
      gap);
}

void PythonTracer::record_return(
    PyFrameObject* frame,
    TracerContext* ctx,
    int type) {
  PyCodeObject* code = PyFrame_GetCode(frame);
  PyObject* co_name = code->co_name;
  const char* name = PyUnicode_AsUTF8(co_name);
  ctx->call_times -= 1;
  unsigned long long s = Timer::getTime() / 1000;
  auto gap = s - ctx->start_time;

  printf(
      "stack = %d, mode = %d, return %s, time = %llu\n",
      ctx->call_times,
      type,
      name,
      gap);
}

} // namespace profiler
} // namespace ftxj