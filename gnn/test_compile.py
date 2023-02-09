import warnings

import torch

# import logging
# from torch._dynamo import optimize, config
# config.verbose=True
# config.log_level = logging.DEBUG
# config.repro_after = "dynamo"
# config.repro_level = 3
# from torch._inductor import config
# config.triton.cudagraphs = False
# config.debug = False
# config.tune_layout = True

from typing import Optional, Tuple

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    else:
        raise ValueError
# Basic "Gather/index-Scatter" patterns commonly used in PyG:
def index_gather(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)

if __name__ == '__main__':
    import argparse

    warnings.filterwarnings('ignore', ".*via the 'torch-scatter' package.*")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()

    num_nodes, num_edges = 10_000, 200_000
    x = torch.randn(num_nodes, 64, device=args.device)
    edge_index = torch.randint(num_nodes, (2, num_edges), device=args.device)
    edge_weight = torch.rand(num_edges, device=args.device)
    matrix = torch.randn(64, 64, device=args.device)
    
    import ctypes
    _cudart = ctypes.CDLL('libcudart.so')
    ret = _cudart.cudaProfilerStart()

    compile_gather_scatter = torch.compile(index_gather)
    reduce = "sum"
    for wramup in range(0, 10):
        index_gather(x, edge_index, reduce)
        compile_gather_scatter(x, edge_index, reduce)

    for test in range(0, 40):
        torch.cuda.nvtx.range_push("eager-" + str(test))
        o1 = index_gather(x, edge_index, reduce)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compile-" + str(test))
        o2 = compile_gather_scatter(x, edge_index, reduce)
        torch.cuda.nvtx.range_pop()
    ret = _cudart.cudaProfilerStop()

    torch.cuda.synchronize()