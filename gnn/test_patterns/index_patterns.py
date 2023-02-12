import torch
from pyg import scatter
from torch import Tensor
import prettytable as pt
import time

from profile_pattern import Pattern, FnWrapper


class IndexselectSubMul(Pattern):
    def __init__(self):
        # Pattern in schnet
        super().__init__()
        self.name = "Schnet::IndexselectSubMul"
        self.shape = [
            [
                [200, 200],
                [200, 200],
                [200, 200],
                [200],
                200,
            ]
        ]

    def init_data(self, test_id):
        h = torch.randn(self.shape[test_id][0], device="cuda")
        c = torch.randn(self.shape[test_id][1], device="cuda")
        pos = torch.randn(self.shape[test_id][2], device="cuda")
        batch = torch.randint(self.shape[test_id][4], self.shape[test_id][3], device="cuda")
        return h, c, pos, batch

    def gen_fn(self):
        x = FnWrapper()
        x.triton_fn = torch.compile(IndexselectSubMul.fn)
        x.nvfuser_fn = torch.jit.script(IndexselectSubMul.fn)
        x.eager_fn = IndexselectSubMul.fn
        self.profile_fn_list.append(x)
    
    # Pattern in schnet
    def fn(h: Tensor, c: Tensor, pos: Tensor, batch: Tensor):
        return h * (pos - torch.index_select(c, 0, batch))

class IndexselectMul(Pattern):
    def __init__(self):
        # Pattern in schnet
        super().__init__()
        self.name = "OC20::IndexselectMul"
        self.shape = [
            [
                [200, 200],
                [200],
                [200, 200],
                200,
            ]
        ]

    def init_data(self, test_id):

        x_kj = torch.randn(self.shape[test_id][0], device="cuda")
        idx_kj = torch.randint(self.shape[test_id][3], self.shape[test_id][1], device="cuda")
        sbf = torch.randn(self.shape[test_id][2], device="cuda")
        return x_kj, idx_kj, sbf

    def gen_fn(self):
        x = FnWrapper()
        x.triton_fn = torch.compile(IndexselectMul.fn)
        x.nvfuser_fn = torch.jit.script(IndexselectMul.fn)
        x.eager_fn = IndexselectMul.fn
        self.profile_fn_list.append(x)
    # Pattern in OC20
    def fn(x_kj: Tensor, idx_kj: Tensor, sbf: Tensor):
        return x_kj[idx_kj] * sbf

class IndexIndexMul(Pattern):
    def __init__(self):
        # Pattern in schnet
        super().__init__()
        self.name = "[P] gcn_norm::Index+Index+Mul"
        self.shape = [ #deg_inv_sqrt, edge_weight, row, col
            [
                [2708], [13264], [13264], [13264], 2707, 2707, # GCNConv, Cora
            ],
            [
                [3327], [12431], [12431], [12431], 3326, 3326, # GCNConv, CiteSeer
            ],
            [
                [19717], [108365], [108365], [108365], 19716, 19716, # GCNConv, CiteSeer
            ],
        ]

    def init_data(self, test_id):

        deg_inv_sqrt = torch.randn(self.shape[test_id][0], device="cuda")
        edge_weight = torch.randn(self.shape[test_id][1], device="cuda")
        row = torch.randint(self.shape[test_id][4], self.shape[test_id][2], device="cuda")
        col = torch.randint(self.shape[test_id][5], self.shape[test_id][3], device="cuda")
        self.str_data = self.shape[test_id][0]
        return deg_inv_sqrt, edge_weight, row, col

    def gen_fn(self):
        pyg = FnWrapper()
        pyg.triton_fn = torch.compile(IndexIndexMul.fn)
        pyg.nvfuser_fn = torch.jit.script(IndexIndexMul.fn)
        pyg.eager_fn = IndexIndexMul.fn
        pyg.name = "(PyG)"
        self.profile_fn_list.append(pyg)
        
        modify = FnWrapper()
        modify.triton_fn = torch.compile(IndexIndexMul.fn_modify)
        modify.nvfuser_fn = torch.jit.script(IndexIndexMul.fn_modify)
        modify.eager_fn = IndexIndexMul.fn_modify
        modify.name = "(modify)"
        self.profile_fn_list.append(modify)


    def fn(deg_inv_sqrt: Tensor, edge_weight: Tensor, row: Tensor, col: Tensor):
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def fn_modify(deg_inv_sqrt: Tensor, edge_weight: Tensor, row: Tensor, col: Tensor):
        return deg_inv_sqrt.index_select(0, row) * edge_weight * deg_inv_sqrt.index_select(0, col)

class ToDenseBatch(Pattern):
    def __init__(self):
        # Pattern in schnet
        super().__init__()
        self.name = "[F] to_dense_batch::elementwise+index"
        self.shape = [ # batch: Tensor, cum_nodes: Tensor, out: Tensor, x: Tensor, max_num_nodes: int
            [
                [1069], [21], [4160, 32], [1069, 32], 2, 19 # mempool, TUD
            ],
            [
                [5031], [129], [64512, 96], [5031, 96], 2, 19 # proteins_gmt, PROTEINS
            ],
        ]

    def init_data(self, test_id):
        self.str_data = self.shape[test_id][0]

        batch = torch.randint(self.shape[test_id][5], self.shape[test_id][0], device="cuda")
        cum_nodes = torch.randint(2, self.shape[test_id][1], device="cuda")
        out = torch.randn(self.shape[test_id][2], device="cuda")
        x = torch.randn(self.shape[test_id][3], device="cuda")
        return batch, cum_nodes, out, x, self.shape[test_id][4]

    def gen_fn(self):
        pyg = FnWrapper()
        pyg.triton_fn = torch.compile(ToDenseBatch.fn)
        pyg.nvfuser_fn = torch.jit.script(ToDenseBatch.fn)
        pyg.eager_fn = ToDenseBatch.fn
        pyg.name = "(PyG)"
        self.profile_fn_list.append(pyg)

        modify = FnWrapper()
        modify.triton_fn = torch.compile(ToDenseBatch.fn_modify)
        modify.nvfuser_fn = torch.jit.script(ToDenseBatch.fn_modify)
        modify.eager_fn = ToDenseBatch.fn_modify
        modify.name = "(modify)"
        self.profile_fn_list.append(modify)


    def fn(batch: Tensor, cum_nodes: Tensor, out: Tensor, x: Tensor, max_num_nodes: int):
        tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes[batch]
        idx = tmp + (batch * max_num_nodes)
        out[idx] = x
        return out

    def fn_modify(batch: Tensor, cum_nodes: Tensor, out: Tensor, x: Tensor, max_num_nodes: int):
        tmp = torch.arange(batch.size(0), device=x.device) - cum_nodes.index_select(0, batch)
        idx = tmp + (batch * max_num_nodes)
        y = out.index_select(0, idx)
        return y

IndexselectSubMul().run()
IndexselectMul().run()
IndexIndexMul().run()
ToDenseBatch().run()
