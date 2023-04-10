#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "tracer.h"

namespace ftxj {
namespace profiler {

struct Interface {
  PyObject_HEAD;
  std::vector<PythonTracer*> tracer_;
  std::vector<CudaTracer*> cu_tracer_;
  bool activate_{false};
};

static PyObject* InterfaceStart(Interface* self, PyObject* args) {
  if (self->activate_) {
    printf("Only a tracer can be activate at a time\n");
    exit(-1);
  }
  self->activate_ = true;
  PythonTracer* tracer = new PythonTracer();
  Py_INCREF(tracer);
  tracer->start();
  self->tracer_.push_back(tracer);
  Py_RETURN_NONE;
}

static PyObject* InterfaceStop(Interface* self, PyObject* args) {
  auto tracer_number = self->tracer_.size();
  if (self->activate_ && tracer_number > 0) {
    self->tracer_[tracer_number - 1]->stop();
  }

  self->activate_ = false;
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

static PyObject* InterfaceEnableCuda(Interface* self) {
  CudaTracer* tracer = new CudaTracer();
  self->cu_tracer_.push_back(tracer);
  tracer->start(self->tracer_[0]->getMeta());
  Py_INCREF(tracer);
  Py_RETURN_NONE;
}

static PyObject* InterfaceDisableCuda(Interface* self) {
  auto tracer_number = self->tracer_.size();
  if (tracer_number > 0) {
    self->tracer_[tracer_number - 1]->stop();
  }
  Py_RETURN_NONE;
}

static PyMethodDef PyInterfaceMethods[] = {
    {"start", (PyCFunction)InterfaceStart, METH_VARARGS, NULL},
    {"stop", (PyCFunction)InterfaceStop, METH_VARARGS, NULL},
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