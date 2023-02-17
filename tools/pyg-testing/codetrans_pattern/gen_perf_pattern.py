from tools.codetrans import Pattern, TransferMethod, to_json

perf_patterns = []

pattern_warm_up = Pattern(
    name = "pattern_warm_up",
    method = TransferMethod.Rewrite,
    old_str = "for epoch in range",
    new_str = '''
import os
warm_up_epoch = os.getenv("WARM_UP_EPOCH")
if(warm_up_epoch):
    warm_up_epoch = eval(warm_up_epoch)
else:
    warm_up_epoch = 1
for epoch in range(1, warm_up_epoch + 2):    
'''
)
perf_patterns.append(pattern_warm_up)


pattern_profile_begin = Pattern(
    name = "pattern_profile_begin",
    method = TransferMethod.Before,
    old_str = "for epoch in range",
    new_str = '''
import ctypes
import time
import os

gpu_profile = os.getenv("FTXJ_GPU_PROF")
if(gpu_profile):
    gpu_profile = eval(gpu_profile)
else:
    gpu_profile = False
_cudart = ctypes.CDLL('libcudart.so')    
'''
)
perf_patterns.append(pattern_profile_begin)


pattern_profile_mid = Pattern(
    name = "pattern_profile_mid",
    method = TransferMethod.After,
    old_str = "for epoch in range",
    new_str = '''
    if(epoch > warm_up_epoch):
        start = time.time()
        if(gpu_profile):
            ret = _cudart.cudaProfilerStart()
            torch.cuda.nvtx.range_push("Profile") 
'''
)
perf_patterns.append(pattern_profile_mid)


pattern_profile_end = Pattern(
    name = "pattern_profile_end",
    method = TransferMethod.AfterBlock,
    old_str = "for epoch in range",
    new_str = '''
if(gpu_profile):
    torch.cuda.nvtx.range_pop()
    ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
end = time.time()
print((end - start) * 1000 / 1)

exit()
'''
)
perf_patterns.append(pattern_profile_end)


to_json(perf_patterns, "./perf.json")
