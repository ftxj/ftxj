#include <Python.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
namespace ftxj {
namespace profiler {

static struct timespec start_time_;

class DataForLog {
 public:
  PyObject* get_name(PyFrameObject* frame, int what, PyObject* arg) {
    PyCodeObject* code = PyFrame_GetCode(frame);
    PyObject* last_name = nullptr;

    if (code) {
      last_name = code->co_name;
    } else {
      last_name = PyUnicode_FromString("<noname>");
    }

    if (what == PyTrace_C_CALL || what == PyTrace_C_RETURN) {
      PyCFunctionObject* cfunc = reinterpret_cast<PyCFunctionObject*>(arg);
      PyObject* cfunc_name = nullptr;
      if (cfunc->m_module) {
        cfunc_name = PyUnicode_FromObject(cfunc->m_module);
        last_name = PyUnicode_Concat(last_name, PyUnicode_FromString("."));
        last_name = PyUnicode_Concat(last_name, cfunc_name);
      }
      if (cfunc->m_ml) {
        cfunc_name = PyUnicode_FromString(cfunc->m_ml->ml_name);
        last_name = PyUnicode_Concat(last_name, PyUnicode_FromString("."));
        last_name = PyUnicode_Concat(last_name, cfunc_name);
      }
    }
    Py_DECREF(code);
    return last_name;
  }
  void push(PyFrameObject* frame, int what, PyObject* arg) {
    if (pointer_ == size_) {
      data_.resize(2 * size_);
      size_ = 2 * size_;
      printf("resize happen = %d\n", size_);
      // exit(1);
    }
    auto name = get_name(frame, what, arg);
    clock_gettime(CLOCK_REALTIME, &data_[pointer_].t_);
    // data_[pointer_].frame = frame;
    data_[pointer_].what = what;
    data_[pointer_].py_name_ = name;
    Py_INCREF(data_[pointer_].py_name_);

    // Py_INCREF(data_[pointer_].frame);
    // Py_INCREF(data_[pointer_].arg);
    pointer_++;
  }

  DataForLog(int stack_size) {
    size_ = stack_size;
    pointer_ = 0;
    data_ = std::vector<PyFrameTrace>(stack_size);
  }
  void print() {
    for (int i = 0; i < pointer_; ++i) {
      data_[i].print();
    }
  }

  PyObject* to_py_obj() {
    PyObject* lst = PyList_New(0);
    for (int i = 0; i < pointer_; ++i) {
      auto dic = data_[i].to_py_dict();
      if (dic)
        PyList_Append(lst, data_[i].to_py_dict());
    }
    return lst;
  }

 private:
  struct PyFrameTrace {
    struct timespec t_;
    int what;
    PyFrameObject* frame;
    PyObject* arg;

    PyObject* py_name_;

    char* name = nullptr;
    unsigned long long ullt_;
    unsigned long long ullt2_;

    void print() {
      unsigned long long t_sec = static_cast<unsigned long long>(t_.tv_sec);
      unsigned long long t_ns = static_cast<unsigned long long>(t_.tv_nsec);
      auto name = (char*)PyUnicode_AsUTF8(py_name_);
      if (what == PyTrace_CALL || what == PyTrace_C_CALL) {
        printf("[%s Call] t = %llu\n", name, t_ns);
      }
      if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
        printf("[%s Return] t = %llu\n", name, t_ns);
      }
      if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
        printf("[%s Return] t = %llu\n", name, t_ns);
      }

      if (what == PyTrace_C_EXCEPTION || what == PyTrace_EXCEPTION) {
        printf("[%s exception] \n", name);
      }

      ullt_ = t_sec * 1000 * 1000 * 1000 + t_ns;
    }
    PyObject* to_py_dict() {
      // if (what != PyTrace_CALL && what != PyTrace_RETURN) {
      //   return nullptr;
      // }
      PyObject* arg_dict = PyDict_New();
      unsigned long long t_sec = static_cast<unsigned long long>(t_.tv_sec);
      unsigned long long t_ns = static_cast<unsigned long long>(t_.tv_nsec);

      unsigned long long t_s1 =
          static_cast<unsigned long long>(start_time_.tv_sec);
      unsigned long long t_s2 =
          static_cast<unsigned long long>(start_time_.tv_nsec);

      ullt_ = t_sec * 1000 * 1000 * 1000 + t_ns;
      ullt2_ = t_s1 * 1000 * 1000 * 1000 + t_s2;

      PyDict_SetItemString(
          arg_dict, "ts", PyFloat_FromDouble((ullt_ - ullt2_) / 1000));
      PyDict_SetItemString(arg_dict, "pid", PyLong_FromLong(0));
      PyDict_SetItemString(arg_dict, "tid", PyLong_FromLong(0));
      if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
        PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("E"));
      }
      if (what == PyTrace_CALL || what == PyTrace_C_CALL) {
        PyDict_SetItemString(arg_dict, "ph", PyUnicode_FromString("B"));
      }

      PyDict_SetItemString(arg_dict, "name", py_name_);
      return arg_dict;
    }
  };

  int size_;
  int pointer_;
  std::vector<PyFrameTrace> data_;
};

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
  void record(PyFrameObject*, TracerContext*, int type, PyObject* arg);
  void record_call(PyFrameObject*, TracerContext*, int type);
  void record_return(PyFrameObject*, TracerContext*, int type);
  void print() {
    logging->print();
  }

  PyObject* toPy() {
    return logging->to_py_obj();
  }

 private:
  bool active_;
  DataForLog* logging;
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
  logging = new DataForLog(10000 * 100);
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

static PyObject* tracer_context_print(TracerContext* self, PyObject* args) {
  self->tracer->print();
  Py_RETURN_NONE;
}

static PyObject* tracer_context_stop(TracerContext* self, PyObject* args) {
  self->tracer->stop(self);
  Py_RETURN_NONE;
}

static PyObject* tracer_context_to_py(TracerContext* self, PyObject* args) {
  return self->tracer->toPy();
}

static PyMethodDef Example_methods[] = {
    {"start",
     (PyCFunction)tracer_context_start,
     METH_VARARGS,
     "tracer_context_start function"},
    {"print",
     (PyCFunction)tracer_context_print,
     METH_VARARGS,
     "tracer_context_print function"},
    {"data",
     (PyCFunction)tracer_context_to_py,
     METH_VARARGS,
     "tracer_context_to_py function"},
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
  clock_gettime(CLOCK_REALTIME, &start_time_);
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
  tracer->record(frame, ctx, what, arg);
  return 0;
}

void PythonTracer::record_call(
    PyFrameObject* frame,
    TracerContext* ctx,
    int type) {
  // logging->push(frame, type);
}

void PythonTracer::record_return(
    PyFrameObject* frame,
    TracerContext* ctx,
    int type) {
  // logging->push(frame, type);
}

void PythonTracer::record(
    PyFrameObject* frame,
    TracerContext* ctx,
    int type,
    PyObject* arg) {
  logging->push(frame, type, arg);
}

} // namespace profiler
} // namespace ftxj