#ifndef __FTXJ_PROFILER_PYTHON_VERSION__
#define __FTXJ_PROFILER_PYTHON_VERSION__
#include <Python.h>
#include <frameobject.h>

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

#endif