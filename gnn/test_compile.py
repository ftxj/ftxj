import warnings

import torch

from torch_geometric.utils import scatter

import logging
from torch._dynamo import optimize, config
config.verbose=True
config.log_level = logging.DEBUG
config.repro_after = "dynamo"
# config.repro_level = 3
# from torch._inductor import config
# config.triton.cudagraphs = False
# config.debug = False
# config.tune_layout = True

# Basic "Gather-Apply-Scatter" patterns commonly used in PyG:
def gather_scatter(x, edge_index, reduce='sum'):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)

def gather_scatter2(x, edge_index):
    x_j = x[edge_index]
    return x_j
    
fn = torch.jit.script(gather_scatter2)
print(fn.graph)
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

    compile_gather_scatter = torch.compile(gather_scatter)
    reduce = "sum"
    for wramup in range(0, 10):
        gather_scatter(x, edge_index, reduce)
        compile_gather_scatter(x, edge_index, reduce)

    for test in range(0, 40):
        torch.cuda.nvtx.range_push("eager-" + str(test))
        o1 = gather_scatter(x, edge_index, reduce)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("compile-" + str(test))
        o2 = compile_gather_scatter(x, edge_index, reduce)
        torch.cuda.nvtx.range_pop()
    ret = _cudart.cudaProfilerStop()

    torch.cuda.synchronize()