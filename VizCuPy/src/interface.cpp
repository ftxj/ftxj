#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "tracer.h"

namespace ftxj {
namespace profiler {

struct Interface {
  PyObject_HEAD;
  std::vector<PythonTracer*> tracer_;
  std::vector<CudaTracer*> cu_tracer_;
  std::vector<MetaEvent*> meta_;
  bool activate_{false};
  bool cu_activate_{false};
  int currect_number{0};
};

namespace impl {
void updateMeta(Interface* self) {
  MetaEvent* meta = new MetaEvent();
  meta->pid = self->meta_.size();
  self->meta_.push_back(meta);
  clock_gettime(CLOCK_REALTIME, &meta->tp_base);
  self->tracer_[self->currect_number]->updateMeta(meta);
  self->cu_tracer_[self->currect_number]->updateMeta(meta);

  self->currect_number = self->meta_.size();
}

void pyTracerStart(Interface* self, bool from_py) {
  if (self->activate_) {
    printf("Only one PyTracer can be activate at a time\n");
    exit(-1);
  }
  self->activate_ = true;
  PythonTracer* tracer = new PythonTracer();
  Py_INCREF(tracer);
  tracer->start(from_py);
  self->tracer_.push_back(tracer);
}

void cudaTracerStart(Interface* self, bool from_py) {
  if (self->cu_activate_) {
    printf("Only one CudaTracer can be activate at a time\n");
    exit(-1);
  }
  self->cu_activate_ = true;
  CudaTracer* tracer = new CudaTracer();
  Py_INCREF(tracer);
  tracer->start(from_py);
  self->cu_tracer_.push_back(tracer);

  updateMeta(self);
}

void pyTracerStop(Interface* self, bool from_py) {
  if (self->activate_ && self->currect_number > 0) {
    self->tracer_[self->currect_number - 1]->stop(from_py);
  }
  self->activate_ = false;
}

void cudaTracerStop(Interface* self, bool from_py) {
  if (self->cu_activate_ && self->currect_number > 0) {
    self->cu_tracer_[self->currect_number - 1]->stop(from_py);
  }
  self->cu_activate_ = false;
}

void tracerStart(Interface* self, bool from_py) {
  pyTracerStart(self, false);
  cudaTracerStart(self, false);
}

void tracerStop(Interface* self, bool from_py) {
  pyTracerStop(self, true);
  cudaTracerStop(self, false);
}

} // namespace impl

static PyObject* InterfaceStart(Interface* self, PyObject* args) {
  impl::tracerStart(self, true);
  Py_RETURN_NONE;
}

static PyObject* InterfaceStop(Interface* self, PyObject* args) {
  impl::tracerStop(self, true);
  Py_RETURN_NONE;
}

static PyObject* InterfaceEnableCuda(Interface* self) {
  impl::cudaTracerStart(self, true);
  Py_RETURN_NONE;
}

static PyObject* InterfaceDisableCuda(Interface* self) {
  impl::cudaTracerStop(self, true);
  Py_RETURN_NONE;
}

static PyObject* InterfaceTimeSplit(Interface* self) {
  if (self->currect_number <= 0) {
    printf("Please Call Split after tracer start!\n");
    printf("If you have done this, this will be a bug.!\n");
    exit(-1);
  }
  impl::pyTracerStop(self, false);
  impl::cudaTracerStop(self, false);
  impl::pyTracerStart(self, false);
  impl::cudaTracerStart(self, false);
  Py_RETURN_NONE;
}

static PyObject* InterfaceNew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  Interface* self = (Interface*)type->tp_alloc(type, 0);
  return (PyObject*)self;
}

static void InterfaceFree(Interface* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* InterfaceDump(Interface* self) {
  PyObject* lst = PyList_New(0);
  for (auto tracer : self->tracer_) {
    PyList_Append(lst, tracer->toPyObj());
  }
  for (auto tracer : self->cu_tracer_) {
    PyList_Append(lst, tracer->toPyObj());
  }
  return lst;
}

static PyMethodDef PyInterfaceMethods[] = {
    {"start", (PyCFunction)InterfaceStart, METH_VARARGS, NULL},
    {"stop", (PyCFunction)InterfaceStop, METH_VARARGS, NULL},
    {"timeline_split", (PyCFunction)InterfaceTimeSplit, METH_VARARGS, NULL},
    {"dump", (PyCFunction)InterfaceDump, METH_VARARGS, NULL},
    {"enable_cuda", (PyCFunction)InterfaceEnableCuda, METH_VARARGS, NULL},
    {"disable_cuda", (PyCFunction)InterfaceDisableCuda, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static PyTypeObject TracerInterfaceType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ftxj.profiler",
    .tp_basicsize = sizeof(Interface),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)InterfaceFree,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Profiler",
    .tp_methods = PyInterfaceMethods,
    .tp_new = InterfaceNew};

static PyMethodDef null_methods[] = {{NULL, NULL, 0, NULL}};

static struct PyModuleDef ProfilerModule =
    {PyModuleDef_HEAD_INIT, "ftxj.profiler", NULL, -1, null_methods};

PyMODINIT_FUNC PyInit_profiler(void) {
  if (PyType_Ready(&TracerInterfaceType) < 0) {
    return NULL;
  }
  PyObject* m = PyModule_Create(&ProfilerModule);
  if (!m) {
    return NULL;
  }
  Py_INCREF(&TracerInterfaceType);
  if (PyModule_AddObject(m, "Tracer", (PyObject*)&TracerInterfaceType) < 0) {
    Py_DECREF(&TracerInterfaceType);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}

} // namespace profiler
} // namespace ftxj