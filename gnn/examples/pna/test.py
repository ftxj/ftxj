import torch

import time
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()
torch._C._jit_set_nvfuser_enabled(True)

x = torch.randn(2,3,4, device='cuda')
i = 2
print(x)

b = x[:, i]

print(b)
# b = torch.randint(0, 1, (1,)).to('cuda')
# b[0] = 1
print(x.index_select(1, torch.IntTensor([i]).to('cuda')).reshape(x.size(0), x.size(2)))
ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()