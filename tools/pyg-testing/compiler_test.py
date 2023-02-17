import torch
from torch import Tensor
from torch_scatter import 
    scatter_max, 
    scatter_sum, 
    scatter_mean, 
    gather_csr,
    segment_csr 

import pytest
import logging
import torch._dynamo as torchdynamo
import torch._dynamo.logging

torchdynamo.config.log_level = logging.CODE
torch._dynamo.config.verbose=True
torch._dynamo.config.suppress_errors = True


@torch._dynamo.optimize("ts_nvfuser")
def dynamo_scatter_max(src, index, dim : int):
    out = torch.add(src, index)
    return out
    
@torch._dynamo.optimize("ts_nvfuser")
def dynamo_scatter_sum(src, index, dim : int):
    out = scatter_sum(src, index, dim=-1)
    out = out + src
    out = out * src
    return out

@torch._dynamo.optimize("ts_nvfuser")
def dynamo_scatter_mean(src, index, dim : int):
    out = scatter_mean(src, index, dim=-1)
    out = out + src
    out = out * src
    return out

@torch._dynamo.optimize("ts_nvfuser")
def dynamo_scatter(src, index, dim : int):
    out = scatter(src, index, dim=1, reduce="sum")
    return out


@torch._dynamo.optimize("ts_nvfuser")
def dynamo_torch_gather(src, index, dim : int):
    out = torch.gather(src, dim, index)
    out = out + src
    out = out * src
    return out

@torch._dynamo.optimize("ts_nvfuser")
def dynamo_gather_csr(src, index):
    out = gather_csr(src, index)
    out = out + src
    out = out * src
    return out

@torch._dynamo.optimize("ts_nvfuser")
def dynamo_gather_csr(src, index):
    out = gather_csr(src, index)
    out = out + src
    out = out * src
    return out


def jit_scatter_max(src, index, dim : int):
    out, argmax = scatter_max(src, index, dim=-1)
    out = out + src
    out = out * src
    return out, argmax
    
def jit_scatter_sum(src, index, dim : int):
    out = scatter_sum(src, index, dim=-1)
    out = out + src
    out = out * src
    return out
    
def jit_scatter_mean(src, index, dim : int):
    out = scatter_mean(src, index, dim=-1)
    out = out + src
    out = out * src
    return out

def jit_scatter(src, index, dim : int):
    out = scatter(src, index, dim=1, reduce="sum")
    return out


def test_scatter_max():
    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 4, 4, 2, 3], [0, 0, 2, 2, 1]])
    

    print("--------------dynamo scatter_max---------------")
    dynamo_scatter_max(src, index, 1)

    fn = torch.jit.script(jit_scatter_max)
    print("--------------script scatter_max---------------")
    print(fn.graph)

    out, argmax = fn(src, index, 1)
    exit()


def test_scatter_sum():
    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 4, 4, 2, 3], [0, 0, 2, 2, 1]])
    
    fn = torch.jit.script(jit_scatter_sum)

    print("--------------script scatter_sum---------------")
    print(fn.graph)

    out = fn(src, index, 1)
    
    print("--------------dynamo scatter_sum---------------")
    dynamo_scatter_sum(src, index, 1)


def test_scatter_mean():
    log = logging.getLogger('test_scatter_max2')
    src = torch.tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
    index = torch.tensor([[4, 4, 4, 2, 3], [0, 0, 2, 2, 1]])
    
    fn = torch.jit.script(jit_scatter_mean)

    print("--------------script scatter_mean---------------")
    print(fn.graph)

    out = fn(src, index, 1)

    print("--------------dynamo scatter_mean---------------")
    dynamo_scatter_mean(src, index, 1)


def main():
    test_scatter_max()
    test_scatter_mean()
    test_scatter_sum()


if __name__ == "__main__":
    main()