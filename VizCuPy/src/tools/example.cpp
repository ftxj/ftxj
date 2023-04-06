#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <frameobject.h>

typedef struct {
  PyObject_HEAD long func1;
} ExampleObject;

#if (                                                                \
    defined(_MSC_VER) && _MSC_VER < 1900 && !defined(__cplusplus) && \
    !defined(inline))
#define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static __inline TYPE
#else
#define PYCAPI_COMPAT_STATIC_INLINE(TYPE) static inline TYPE
#endif

#ifndef _Py_CAST
#define _Py_CAST(type, expr) ((type)(expr))
#endif

#if PY_VERSION_HEX < 0x030A00A3 && !defined(Py_NewRef)
PYCAPI_COMPAT_STATIC_INLINE(PyObject*)
_Py_NewRef(PyObject* obj) {
  Py_INCREF(obj);
  return obj;
}
#define Py_NewRef(obj) _Py_NewRef(_PyObject_CAST(obj))
#endif

#if PY_VERSION_HEX < 0x030900B1
PYCAPI_COMPAT_STATIC_INLINE(PyCodeObject*)
PyFrame_GetCode(PyFrameObject* frame) {
  assert(frame != _Py_NULL);
  assert(frame->f_code != _Py_NULL);
  return _Py_CAST(PyCodeObject*, Py_NewRef(frame->f_code));
}
#endif

int frame_tarce_function(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  PyCodeObject* code = PyFrame_GetCode(frame);
  PyObject* co_name = code->co_name;
  const char* name = PyUnicode_AsUTF8(co_name);
  if (what == PyTrace_CALL) {
    printf("call %s\n", name);
  }
  if (what == PyTrace_RETURN) {
    printf("return %s\n", name);
  }
  Py_XDECREF(code);
  return 0;
}

int disable_frame_trace(
    PyObject* obj,
    PyFrameObject* frame,
    int what,
    PyObject* arg) {
  return 0;
}

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
    {PyModuleDef_HEAD_INIT, "ftxj.profiler", NULL, -1, example_methods};

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
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "ftxj.profiler",
    .tp_basicsize = sizeof(ExampleObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Example_delete,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Example",
    .tp_methods = Example_methods,
    .tp_new = Example_New};

PyMODINIT_FUNC PyInit_profiler(void) {
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