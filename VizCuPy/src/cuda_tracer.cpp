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
  struct timespec tp_start_;
  struct timespec tp_end_;

  CudaTracerLocalResult(const size_t& size) {
    record_ = new RecordQueue(size);
  }

  PyObject* toPyObj() {
    return record_->toPyObj();
  }
};

namespace detail {

void traceRuntimeAPI(
    CudaTracer* tracer,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  static constexpr auto E1 = EventType::CudaCall;
  static constexpr auto E2 = EventType::CudaReturn;
  EventType E;
  if (cbInfo->callbackSite == CUPTI_API_ENTER) {
    E = E1;
    switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
        tracer->recordCudaCall(cbInfo->symbolName);
        break;
      case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
        tracer->recordCudaCall(cbInfo->functionName);
        break;
      default:
        break;
    }
  } else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
    E = E2;
    switch (cbid) {
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
      case CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020:
      case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
        tracer->recordCudaReturn();
        break;
      default:
        break;
    }
  } else {
    printf("Unkown cbInfo type\n");
  }
}

void CUPTIAPI getTimestampCallback(
    void* userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const CUpti_CallbackData* cbInfo) {
  CudaTracer* tracer = static_cast<CudaTracer*>(userdata);
  switch (domain) {
    case CUPTI_CB_DOMAIN_DRIVER_API:
      printf("driver api\n");
      break;
    case CUPTI_CB_DOMAIN_RUNTIME_API:
      traceRuntimeAPI(tracer, cbid, cbInfo);
      break;
    default:
      printf("unknow API when tracing %d\n", domain);
  }
}

} // namespace detail

static CUpti_SubscriberHandle subscriber;

CudaTracer::CudaTracer() {}

void CudaTracer::start() {
  ASSERT(activate_ == false, "CudaTracer start twice\n");
  activate_ = true;
  local_results_ = new CudaTracerLocalResult(queue_size);
  std::vector<PyFrameObject*> current_stack;
  CUPTI_CALL(cuptiSubscribe(
      &subscriber, (CUpti_CallbackFunc)detail::getTimestampCallback, this));
  DRIVER_API_CALL(cuInit(0));
  CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
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

void CudaTracer::recordCudaCall(const char* cuda_name) {
  static constexpr auto E = EventType::CudaCall;
  EventTag T = EventTag::Cuda;
  if (local_results_ == nullptr || local_results_->record_ == nullptr) {
    printf("bug happen, we doesn't have queue\n");
    exit(-1);
  }
  Event* event = local_results_->record_->getNextNeedRecorded();
  if (event == nullptr) {
    printf("bug happen, we doesn't return event\n");
    exit(-1);
  }
  event->caller_ = father;
  event->type_ = E;
  event->tag_ = T;
  event->recordName(cuda_name);
  event->recordTime();
  father = event;
  call_stack_depth++;
}

void CudaTracer::recordCudaReturn() {
  static constexpr auto E = EventType::CudaComplete;
  if (father == nullptr || call_stack_depth <= 0) {
    printf("bug happen, cpp call stack record fail\n");
    exit(-1);
  }
  call_stack_depth--;
  father->recordDuration();
  father->type_ = E;
  father = father->caller_;
}

PyObject* CudaTracer::toPyObj() {
  return local_results_->toPyObj();
}

} // namespace profiler
} // namespace ftxj