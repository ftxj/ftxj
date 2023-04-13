#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "timeline_schedule.h"
#include "tracer.h"
namespace ftxj {
namespace profiler {

struct Interface {
  PyObject_HEAD;
  PythonTracer* tracer_;
  // CudaTracer* cu_tracer_;
  TimeLineSchedule* timeline_schedule_;
  bool activate_{false};
  // bool cu_activate_{false};
  struct timespec tp_start_;
};

namespace impl {

void pyTracerStart(Interface* self) {
  if (self->activate_) {
    printf("Only one PyTracer can be activate at a time\n");
    exit(-1);
  }
  self->activate_ = true;
  self->tracer_ = new PythonTracer();
  Py_INCREF(self->tracer_);

  clock_gettime(CLOCK_REALTIME, &(self->tp_start_));

  self->tracer_->start(&(self->tp_start_));
  self->timeline_schedule_ = new TimeLineSchedule(self->tracer_->getQueue());
}

// void cudaTracerStart(Interface* self, bool from_py) {
//   if (self->cu_activate_) {
//     printf("Only one CudaTracer can be activate at a time\n");
//     exit(-1);
//   }
//   self->cu_activate_ = true;
//   CudaTracer* tracer = new CudaTracer();
//   Py_INCREF(tracer);
//   tracer->start(from_py);
//   self->cu_tracer_.push_back(tracer);

//   updateMeta(self);
// }

void pyTracerStop(Interface* self) {
  self->tracer_->stop();
  self->activate_ = false;
}

// void cudaTracerStop(Interface* self, bool from_py) {
//   if (self->cu_activate_ && self->currect_number > 0) {
//     self->cu_tracer_[self->currect_number - 1]->stop(from_py);
//   }
//   self->cu_activate_ = false;
// }

void tracerStart(Interface* self) {
  pyTracerStart(self);
}

void tracerStop(Interface* self) {
  pyTracerStop(self);
}

} // namespace impl

static PyObject* InterfaceStart(Interface* self, PyObject* args) {
  impl::tracerStart(self);
  Py_RETURN_NONE;
}

static PyObject* InterfaceStop(Interface* self, PyObject* args) {
  impl::tracerStop(self);
  Py_RETURN_NONE;
}

// static PyObject* InterfaceEnableCuda(Interface* self) {
//   impl::cudaTracerStart(self, true);
//   Py_RETURN_NONE;
// }

// static PyObject* InterfaceDisableCuda(Interface* self) {
//   impl::cudaTracerStop(self, true);
//   Py_RETURN_NONE;
// }

static PyObject* InterfaceTimeSplit(Interface* self) {
  if (self->activate_ == false) {
    printf("Please Call Split after tracer start!\n");
    printf("If you have done this, this will be a bug.!\n");
    exit(-1);
  }
  self->timeline_schedule_->split();
  Py_RETURN_NONE;
}

static PyObject* InterfaceTimeSync(Interface* self) {
  if (self->activate_ == false) {
    printf("Please Call Split after tracer start!\n");
    printf("If you have done this, this will be a bug.!\n");
    exit(-1);
  }
  self->timeline_schedule_->align();
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
  // PyObject* lst = PyList_New(0);
  if (self == nullptr || self->timeline_schedule_ == nullptr) {
    printf("bug happen in dump\n");
    exit(-1);
  }
  self->timeline_schedule_->update_queue_data(self->tp_start_);
  // PyList_Append(lst, self->tracer_->toPyObj());
  return self->tracer_->toPyObj();
}

static PyMethodDef PyInterfaceMethods[] = {
    {"start", (PyCFunction)InterfaceStart, METH_VARARGS, NULL},
    {"stop", (PyCFunction)InterfaceStop, METH_VARARGS, NULL},
    {"timeline_split", (PyCFunction)InterfaceTimeSplit, METH_VARARGS, NULL},
    {"timeline_sync", (PyCFunction)InterfaceTimeSync, METH_VARARGS, NULL},
    {"dump", (PyCFunction)InterfaceDump, METH_VARARGS, NULL},
    // {"enable_cuda", (PyCFunction)InterfaceEnableCuda, METH_VARARGS, NULL},
    // {"disable_cuda", (PyCFunction)InterfaceDisableCuda, METH_VARARGS, NULL},
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