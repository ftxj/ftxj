import torch
from pyg import scatter
import time
from torch import Tensor
import prettytable as pt
torch._C._jit_set_nvfuser_enabled(True)


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


eager_fn_list = [
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

def benchmark(fn, name, *args):
    for i in range(10):
        fn(*args)

    torch.cuda.synchronize()
    start = time.time()
    for i in range(30):
        fn(*args)
    torch.cuda.synchronize()

    stop = time.time()
    time.sleep(1)
    return stop - start

def check(eager, nvfuser, triton, *args):
    o1 = nvfuser(*args)
    o2 = triton(*args)
    e = eager(*args)
    assert torch.allclose(o1, e, atol=1e-4)
    assert torch.allclose(o2, e, atol=1e-4)

    
if __name__ == '__main__':

    data = [
        [100, 200, 4],
        [10_000, 200_000, 64],
        [10_000, 2_000_000, 64],
        [100_000, 2_000_000, 32],
    ]

    summary = pt.PrettyTable(["(Node, Edge, Feature)", "Pattern", "eager", "triton", "nvfuser", "triton/nvfuser"])


    triton_fn_list = []
    nvfuser_fn_list = []

    for fn in eager_fn_list:
        if(fn != index_select_elementwise_reduce):
            triton_fn_list.append(torch.compile(fn))
        nvfuser_fn_list.append(torch.jit.script(fn))
        # print(nvfuser_fn_list[-1].graph)
        
    with torch.jit.fuser("fuser2"):
        for idx in range(0, len(eager_fn_list)):

            for data_idx in range(0, len(data)):
                
                num_nodes, num_edges, feature = data[data_idx]
                str_data = str(num_nodes) + ", " + str(num_edges) + ", " + str(feature)
                x = torch.randn(num_nodes, feature, device="cuda")
                edge_index = torch.randint(num_nodes, (2, num_edges), device="cuda")
                row, col = edge_index

                t_eager = benchmark(eager_fn_list[idx], "eager", x, row, col)
                
                t_triton = "None"
                if(eager_fn_list[idx] != index_select_elementwise_reduce):
                    t_triton = benchmark(triton_fn_list[idx], "triton", x, row, col)
                t_nvfuser = benchmark(nvfuser_fn_list[idx], "nvfuser", x, row, col)
                

                if(t_triton != "None"):
                    check(eager_fn_list[idx],nvfuser_fn_list[idx],triton_fn_list[idx], x, row, col)
                    summary.add_row([str_data, test_names[idx], 1, str(round(t_eager / t_triton, 2))+"x", str(round(t_eager / t_nvfuser, 2)) + "x", str(round(t_triton / t_nvfuser, 2)) + "x"])
                else:
                    check(eager_fn_list[idx],nvfuser_fn_list[idx],eager_fn_list[idx], x, row, col)
                    summary.add_row([str_data, test_names[idx], 1, "Fail", str(round(t_eager / t_nvfuser, 2)) + "x", "---"])
    

    print(summary)
    

   