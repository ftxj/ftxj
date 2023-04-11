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
  clock_gettime(CLOCK_REALTIME, &meta->tp_base);
  meta->pid = self->meta_.size();
  meta->tid = 0;
  self->meta_.push_back(meta);
  self->currect_number = self->meta_.size();
}

void pyTracerStart(Interface* self) {
  if (self->activate_) {
    printf("Only one PyTracer can be activate at a time\n");
    exit(-1);
  }
  self->activate_ = true;
  updateMeta(self);
  PythonTracer* tracer = new PythonTracer();
  Py_INCREF(tracer);
  tracer->start(self->meta_[self->currect_number - 1]);
  self->tracer_.push_back(tracer);
}

void cudaTracerStart(Interface* self) {
  if (self->cu_activate_) {
    printf("Only one CudaTracer can be activate at a time\n");
    exit(-1);
  }
  self->cu_activate_ = true;
  CudaTracer* tracer = new CudaTracer();
  Py_INCREF(tracer);
  tracer->start(self->meta_[self->currect_number - 1]);
  self->cu_tracer_.push_back(tracer);
}

void pyTracerStop(Interface* self) {
  if (self->activate_ && self->currect_number > 0) {
    self->tracer_[self->currect_number - 1]->stop();
  }
  self->activate_ = false;
}

void cudaTracerStop(Interface* self) {
  if (self->cu_activate_ && self->currect_number > 0) {
    self->cu_tracer_[self->currect_number - 1]->stop();
  }
  self->cu_activate_ = false;
}

} // namespace impl

static PyObject* InterfaceStart(Interface* self, PyObject* args) {
  impl::pyTracerStart(self);
  Py_RETURN_NONE;
}

static PyObject* InterfaceStop(Interface* self, PyObject* args) {
  impl::pyTracerStop(self);
  Py_RETURN_NONE;
}

static PyObject* InterfaceEnableCuda(Interface* self) {
  impl::cudaTracerStart(self);
  Py_RETURN_NONE;
}

static PyObject* InterfaceDisableCuda(Interface* self) {
  impl::cudaTracerStop(self);
  Py_RETURN_NONE;
}

static PyObject* InterfaceTimeSplit(Interface* self) {
  if (self->currect_number <= 0) {
    printf("Please Call Split after tracer start!\n");
    printf("If you have done this, this will be a bug.!\n");
    exit(-1);
  }
  impl::pyTracerStop(self);
  impl::cudaTracerStop(self);
  impl::pyTracerStart(self);
  impl::cudaTracerStart(self);
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