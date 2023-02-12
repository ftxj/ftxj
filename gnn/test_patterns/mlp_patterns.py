import torch
from pyg import scatter
from torch import Tensor
import prettytable as pt
import time

from profile_pattern import Pattern, FnWrapper

from torch_geometric.nn import MLP

class MLPBlock(Pattern):
    def __init__(self):
        # Pattern in schnet
        super().__init__()
        self.name = "[F] rgcn::index+gather+elementwise"
        self.shape = [ # tmp: Tensor, index: Tensor, edge_type: Tensor, max1, max2
            [
                [6909, 90], [54024], [54024], [54024, 4], 6908, 87, # rgcn, AIFB
            ],
            [
                [23606, 46], [148082], [148082], [148082, 2], 23605, 45 # rgcn, MUTAG
            ]
        ]

    def init_data(self, test_id):
        self.str_data = self.shape[test_id][0]
        self.mlp = 
        tmp = torch.randn(self.shape[test_id][0], device="cuda")
        index = torch.randint(self.shape[test_id][4], self.shape[test_id][1], device="cuda")
        edge_type = torch.randint(self.shape[test_id][5], self.shape[test_id][2], device="cuda")
        indexs = torch.randn(self.shape[test_id][3], device="cuda")
        return tmp, index, edge_type, indexs

    def gen_fn(self):
        pyg = FnWrapper()
        pyg.triton_fn = torch.compile(GatherElementwise.fn)
        pyg.nvfuser_fn = torch.jit.script(GatherElementwise.fn)
        pyg.eager_fn = GatherElementwise.fn
        pyg.name = "(PyG)"
        self.profile_fn_list.append(pyg)

        modify = FnWrapper()
        modify.triton_fn = torch.compile(GatherElementwise.fn_modify)
        modify.nvfuser_fn = torch.jit.script(GatherElementwise.fn_modify)
        modify.eager_fn = GatherElementwise.fn_modify
        modify.name = "(modify)"
        self.profile_fn_list.append(modify)


GatherElementwise().run()
