#ifndef __FTXJ_PROFILER_TRACER_CONTEXT__
#define __FTXJ_PROFILER_TRACER_CONTEXT__

#include <Python.h>
#include <frameobject.h>

struct PythonTracer;

struct TracerContext {
  PyObject_HEAD;
  PythonTracer* tracer;
  int call_times;
};

static PyObject* start();
static PyObject* stop();

static PyTypeObject TracerContextType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ftxj.profiler",
    .tp_doc = "Profiler",
    .tp_basicsize = sizeof(TracerContext),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Example_New,
    .tp_dealloc = (destructor)Example_delete,
    .tp_methods = Example_methods};

#endif