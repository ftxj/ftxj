#include <Python.h>
#include <cuda.h>
#include <cupti.h>
#include <frameobject.h>
#include <ctime>
#include <vector>
#include "cpython_version.h"
#include "event.h"
#include "queue.h"
#include "tracer.h"
#include "util.h"

namespace ftxj {
namespace profiler {

#define DRIVER_API_CALL(apiFuncCall)                           \
  do {                                                         \
    CUresult _status = apiFuncCall;                            \
    if (_status != CUDA_SUCCESS) {                             \
      const char* errstr;                                      \
      cuGetErrorString(_status, &errstr);                      \
      fprintf(                                                 \
          stderr,                                              \
          "%s:%d: error: function %s failed with error %s.\n", \
          __FILE__,                                            \
          __LINE__,                                            \
          #apiFuncCall,                                        \
          errstr);                                             \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                          \
  do {                                                         \
    cudaError_t _status = apiFuncCall;                         \
    if (_status != cudaSuccess) {                              \
      fprintf(                                                 \
          stderr,                                              \
          "%s:%d: error: function %s failed with error %s.\n", \
          __FILE__,                                            \
          __LINE__,                                            \
          #apiFuncCall,                                        \
          cudaGetErrorString(_status));                        \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (0)

#define CUPTI_CALL(call)                                       \
  do {                                                         \
    CUptiResult _status = call;                                \
    if (_status != CUPTI_SUCCESS) {                            \
      const char* errstr;                                      \
      cuptiGetResultString(_status, &errstr);                  \
      fprintf(                                                 \
          stderr,                                              \
          "%s:%d: error: function %s failed with error %s.\n", \
          __FILE__,                                            \
          __LINE__,                                            \
          #call,                                               \
          errstr);                                             \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  } while (0)

struct CudaTracerLocalResult {
  RecordQueue* record_{nullptr};
  struct timespec t_s_;
  struct timespec t_e_;
  CudaTracerLocalResult(const size_t& size, const MetaEvent* m) {
    record_ = new RecordQueue(size, m);
  }
};

namespace detail {

static PyObject* toPyStr(const char* str) {
  auto name = PyUnicode_FromString(str);
  return name;
}

void traceRuntimeAPI(
    CudaTracerLocalResult* data,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  static constexpr auto E1 = EventType::CudaCall;
  static constexpr auto E2 = EventType::CudaReturn;
  EventType E;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    E = E1;
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    E = E2;
  } else {
    printf("Unkown cbInfo type\n");
  }
  switch (cbid) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      data->record_->record(E, toPyStr(cbInfo->symbolName));
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
      printf("sync kernel %s\n", cbInfo->functionName);
      break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
      printf("cuda memcpy\n", cbInfo->functionName);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuDeviceGet:
    case CUPTI_DRIVER_TRACE_CBID_cuCtxSynchronize:
    case CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize_ptsz:
    case CUPTI_DRIVER_TRACE_CBID_cuCtxCreate:
    case CUPTI_DRIVER_TRACE_CBID_cuMemPeerGetDevicePointer:
    case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
    case CUPTI_DRIVER_TRACE_CBID_cu64MemHostGetDevicePointer:
    case CUPTI_DRIVER_TRACE_CBID_cu64GraphicsResourceGetMappedPointer:
      break;
    default:
      printf("unknow API when tracing %d\n", cbid);
      exit(-1);
  }
}

void CUPTIAPI getTimestampCallback(
    void* userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  CudaTracerLocalResult* traceData = (CudaTracerLocalResult*)userdata;
  switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API:
      printf("driver api\n");
      break;
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      traceRuntimeAPI(traceData, cbid, cbInfo);
      break;
    default:
      printf("unknow API when tracing %d\n", domain);
  }
}

} // namespace detail

static CUpti_SubscriberHandle subscriber;

CudaTracer::CudaTracer() {}

void CudaTracer::start(MetaEvent* m) {
  if (!activate_) {
    meta = new MetaEvent();
    meta->tp_base = m->tp_base;
    meta->pid = m->pid;
    meta->tid = 1;
    local_results_ = new CudaTracerLocalResult(100000, meta);
    activate_ = true;
    CUPTI_CALL(cuptiSubscribe(
        &subscriber,
        (CUpti_CallbackFunc)detail::getTimestampCallback,
        local_results_));
    DRIVER_API_CALL(cuInit(0));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  }
}

void CudaTracer::stop() {
  if (activate_) {
    CUPTI_CALL(cuptiUnsubscribe(subscriber));
    activate_ = false;
  }
}

CudaTracer::~CudaTracer() {
  delete local_results_;
}

PyObject* CudaTracer::toPyObj() {
  return local_results_->record_->toPyObj();
}

} // namespace profiler
} // namespace ftxj