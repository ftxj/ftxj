import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import logging
from torch._dynamo import optimize, config
# config.verbose=True
# config.log_level = logging.DEBUG
# # config.repro_after = "dynamo"
# # config.repro_level = 3
# # from torch._inductor import config
# # config.triton.cudagraphs = False
# # config.debug = False
# # config.tune_layout = True

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()




class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, idx, src):
        return torch.scatter_add(inp, 0, idx, src)
        


model = Net().to('cuda')

model = torch.compile(model)

inp = torch.randn(32, 32, device="cuda", dtype=torch.float)
idx = torch.randint(0, 32, (16, 16), device="cuda").to(dtype=torch.long)
src = torch.randn(64, 32, device="cuda", dtype=torch.float)

for i in range(20):
    out = model(inp, idx, src)
    out2 = torch.scatter_add(inp, 0, idx, src)

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()