import math

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Parameter

from torch_geometric.typing import Adj, OptTensor, SparseTensor, Optional


class NetBase2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    @torch.jit._overload_method
    def foo(self, t1, t2):
        # type: (Tensor, Tensor) -> Tensor
        pass

    @torch.jit._overload_method
    def foo(self, t1, t2):
        # type: (SparseTensor, SparseTensor) -> SparseTensor
        pass

    @torch.jit.export
    def foo(self, t1: Adj, t2: Adj):
        return t1

class NetBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # @torch.jit._overload_method
    # def forward(self, edge_index: SparseTensor):
    #     # type: (SparseTensor) -> int
    #     pass

    # @torch.jit._overload_method
    # def forward(self, edge_index: Tensor):
    #     # type: (Tensor) -> int
    #     pass

    def forward(self, edge_index: Adj) -> int:
        if isinstance(edge_index, SparseTensor):
            return 30
        else:
            return 50

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = NetBase()

    @torch.jit._overload_method
    def forward(self, edge_index: SparseTensor):
        # type: (SparseTensor) -> int
        pass

    @torch.jit._overload_method
    def forward(self, edge_index: Tensor):
        # type: (Tensor) -> int
        pass

    def forward(self, edge_index: Adj) -> int:
        return self.b(edge_index)



class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = NetBase2()

    @torch.jit._overload_method
    def forward(self, t1, t2):
        # type: (SparseTensor, SparseTensor) -> SparseTensor
        pass

    @torch.jit._overload_method
    def forward(self, t1, t2):
        # type: (Tensor, Tensor) -> Tensor
        pass

    def forward(self, t1: Adj, t2: Adj):
        return t1

    @torch.jit._overload_method
    def f2(self, t3, t4):
        # type: (SparseTensor, SparseTensor) -> SparseTensor
        pass

    @torch.jit._overload_method
    def f2(self, t3, t4):
        # type: (Tensor, Tensor) -> Tensor
        pass

    @torch.no_grad()
    def f2(self, t3: Adj, t4: Adj):
        return self.b.foo(t3,t4)
    

    def jittable(self):
        class wrapper(torch.nn.Module):
            def __init__(self, child):
                super().__init__()
                self.m = child

            def forward(self, t1, t2):
                return self.m(t1,t2)

            @torch.jit.export
            def f2(self, t3: Tensor, t4: Tensor):
                return self.m.f2(t3,t4)
        return wrapper(self)