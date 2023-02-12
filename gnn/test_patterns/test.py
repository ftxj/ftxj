import torch
from pyg import scatter
import time
from torch import Tensor
import prettytable as pt
import os
from contextlib import contextmanager

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()

torch._C._jit_set_nvfuser_enabled(True)

@contextmanager
def graph_op_fusion():
    os.environ['PYTORCH_NVFUSER_ENABLE'] = "graph_op_fusion"
    yield
    os.environ['PYTORCH_NVFUSER_ENABLE'] = ""


def index_scatter(x, row, col):
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0))

def index_select_scatter_pyg(x: Tensor, row, col):
    x_j = torch.index_select(x, 0, row)
    return scatter(x_j, col, dim_size=x.size(0))

def index_select_scatter_modifyed(x: Tensor, row, col):
    out = torch.zeros(x.size(), device='cuda')
    x_j = torch.index_select(x, 0, row)
    idx = col.unsqueeze(-1)
    return torch.scatter_add(out, 0, idx, x_j)



def index_cat_scatter(x: Tensor, row, col):
    x_ij = torch.cat([x[col], x[row]], dim=-1)
    return scatter(x_ij, col, dim_size=x.size(0))

def index_select_cat_scatter(x: Tensor, row, col):
    x_ij = torch.cat([torch.index_select(x, 0, row), torch.index_select(x, 0, col)], dim=-1)
    return scatter(x_ij, col, dim_size=x.size(0))

def index_select_cat_scatter_modify(x: Tensor, row, col):
    out = torch.zeros(x.size(), device='cuda')
    idx = col.unsqueeze(-1)
    x_ij = torch.cat([torch.index_select(x, 0, row), torch.index_select(x, 0, col)], dim=-1)
    return torch.scatter_add(out, 0, idx, x_ij)


# def gather_elementwise(x: Tensor, row, col):
#     out = torch.zeros([len(row), len(row)], device='cuda')
#     tmp = torch.gather(out, 1, row.unsqueeze(-1))
#     return 1. / torch.clamp(tmp, 1.)

def index_select_elementwise_reduce(x: Tensor, row, col):
    x_i = torch.index_select(x, 0, row)
    x_j = torch.index_select(x, 0, col)
    return torch.sum(x_i + x_j, -1)


# eager_fn_list = [
#     [index_scatter, "index + scatter (PyG)", ["triton", "nvfuser"]],
#     [index_select_scatter_pyg, "index_select + scatter(little Modify)", ["triton", "nvfuser"]],
#     [index_select_scatter_modifyed, "index_select + scatter(lots Modify)", ["triton", "nvfuser"]],
#     [index_cat_scatter, "index + cat + scatter(PyG)", ["triton", "nvfuser"]],
#     [index_select_cat_scatter, "index_select + cat + scatter(little Modify)", ["triton", "nvfuser"]],
#     [index_select_cat_scatter_modify, "index_select + cat + scatter(lots Modify)", ["triton", "nvfuser"]],
#     [index_select_elementwise_reduce, "index_select + add + sum", ["triton", "nvfuser"]]
# ]

test_fns = [
    index_scatter,
    index_select_scatter_pyg, 
    index_select_scatter_modifyed,
    index_cat_scatter,
    index_select_cat_scatter,
    index_select_cat_scatter_modify,
    index_select_elementwise_reduce
]


test_names = [
    "index + scatter (PyG)",
    "index_select + scatter(little Modify)", 
    "index_select + scatter(lots Modify)",
    "index + cat + scatter(PyG)", 
    "index_select + cat + scatter(little Modify)",
    "index_select + cat + scatter(lots Modify)",
    "index_select + add + sum"
]

triton_fail_list = [
    "index_select + add + sum"
]

def benchmark(fn, name, *args):
    if fn == None:
        return None

    torch.cuda.nvtx.range_push("warmup")
    for i in range(10):
        fn(*args)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(name)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(30):
        fn(*args)
    torch.cuda.synchronize()
    stop = time.time()
    torch.cuda.nvtx.range_pop()

    time.sleep(1)
    return stop - start

def check(eager, nvfuser, triton, *args):
    e = eager(*args)
    if(nvfuser != None):
        o = nvfuser(*args)
        assert torch.allclose(o, e, atol=1e-4)
    if(triton != None):
        o = triton(*args)
        assert torch.allclose(o, e, atol=1e-4)

def speedups(eager, nvfuser, triton):
    triton_eager, nvfuser_eager, nvfuser_triton = "/", "/", "/"
    if nvfuser != None:
        nvfuser_eager = str(round(eager / nvfuser, 2)) + "x"
    if triton != None:
        triton_eager = str(round(eager / triton, 2)) + "x"
    if triton != None and nvfuser != None:
        nvfuser_triton = str(round(triton / nvfuser, 2)) + "x"
    return triton_eager, nvfuser_eager, nvfuser_triton

if __name__ == '__main__':

    graph_data = [
        [100, 200, 4],
        [10_000, 200_000, 64],
        [10_000, 2_000_000, 64],
        [100_000, 2_000_000, 32],
    ]

    summary = pt.PrettyTable(["(Node, Edge, Feature)", "Pattern", "eager", "triton", "nvfuser", "nvfuser vs triton"])

    eager_fn_list = {}
    triton_fn_list = {}
    nvfuser_fn_list = {}
    nvfuser_graph_fn_list = {}

    for fn, name in zip(test_fns, test_names):
        eager_fn_list[name] = fn

        if name in triton_fail_list:
            triton_fn_list[name] = None
        else:
            triton_fn_list[name] = torch.compile(fn)

        nvfuser_fn_list[name] = torch.jit.script(fn)
        
    with torch.jit.fuser("fuser2"):
        for fn, name in zip(test_fns, test_names):
            for data in graph_data:

                num_nodes, num_edges, feature = data
                str_data = str(num_nodes) + ", " + str(num_edges) + ", " + str(feature)
                x = torch.randn(num_nodes, feature, device="cuda")
                edge_index = torch.randint(num_nodes, (2, num_edges), device="cuda")
                row, col = edge_index

                t_eager = benchmark(eager_fn_list[name], "eager-" + name, x, row, col)
                t_triton = benchmark(triton_fn_list[name], "triton-" + name, x, row, col)
                t_nvfuser = benchmark(nvfuser_fn_list[name], "nvfuser-" + name, x, row, col)

                check(eager_fn_list[name],nvfuser_fn_list[name],triton_fn_list[name], x, row, col)

                t1, t2, t3 = speedups(t_eager, t_nvfuser, t_triton)
                summary.add_row([str_data, name, 1, t1, t2, t3])
    
    print(summary)
    
ret = _cudart.cudaProfilerStop()
   