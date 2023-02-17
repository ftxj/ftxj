from ts_test import Net2
import torch

x = torch.jit.script(Net2().jittable())

print(x(torch.randn(2,2), torch.randn(2,2)))
print(x.f2(torch.randn(2,2), torch.randn(2,2)))
