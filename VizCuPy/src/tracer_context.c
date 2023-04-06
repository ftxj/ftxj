#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>
#include "cpython_version.h"
#include "tracer_context.h"

static PyObject* getvar1(ExampleObject* self, PyObject* args) {
  PyEval_SetProfile(NULL, NULL);
  return PyLong_FromLong(self->func1);
}

static PyObject* setvar1(ExampleObject* self, PyObject* args) {
  PyArg_ParseTuple(args, "l", &self->func1);
  PyEval_SetProfile(frame_tarce_function, NULL);
  Py_RETURN_NONE;
}

static PyMethodDef Example_methods[] = {
    {"get", (PyCFunction)getvar1, METH_VARARGS, "getvar1 function"},
    {"set", (PyCFunction)setvar1, METH_VARARGS, "setvar1 function"},
    {NULL, NULL, 0, NULL}};

static PyMethodDef example_methods[] = {{NULL, NULL, 0, NULL}};

static struct PyModuleDef example_module =
    {PyModuleDef_HEAD_INIT, "vizcupy.example", NULL, -1, example_methods};

static PyObject* Example_New(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  printf("new example object\n");
  ExampleObject* self = (ExampleObject*)type->tp_alloc(type, 0);
  if (self) {
    self->func1 = 0;
  }
  return (PyObject*)self;
}

static void Example_delete(ExampleObject* self) {
  self->func1 = 1;
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyTypeObject ExampleType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "VizCuPy.Example",
    .tp_doc = "Example",
    .tp_basicsize = sizeof(ExampleObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Example_New,
    .tp_dealloc = (destructor)Example_delete,
    .tp_methods = Example_methods};

PyMODINIT_FUNC PyInit_example(void) {
  printf("init py example\n");
  if (PyType_Ready(&ExampleType) < 0) {
    return NULL;
  }
  PyObject* m = PyModule_Create(&example_module);
  if (!m) {
    return NULL;
  }
  Py_INCREF(&ExampleType);
  if (PyModule_AddObject(m, "Example", (PyObject*)&ExampleType) < 0) {
    Py_DECREF(&ExampleType);
    Py_DECREF(m);
    return NULL;
  }
  return m;
}