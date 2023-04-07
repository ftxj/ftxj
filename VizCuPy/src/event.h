
#ifndef __EVENTNODE_H__
#define __EVENTNODE_H__

#include <Python.h>
#include <string>

class EventBase {};

using time_gap_t = unsigned long long;

class PyFunctionEvent : public EventBase {
  static std::string category_name = "Python Function";
};

class CudaFunctionEvent : public EventBase {};

#endif