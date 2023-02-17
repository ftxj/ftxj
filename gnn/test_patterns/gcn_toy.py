import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter(src, index, dim, dim_size):
    index = broadcast(index, src, dim)
    size = list(src.size())
    size[dim] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    return out.scatter_add_(dim, index, src)

class GCN(torch.nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int):
        super().__init__()
        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.linear(x, self.weight, self.bias)

        x_j = x.index_select(0, edge_index[0])
        out = self.message(x_j)
        out = self.aggregate(out, edge_index[1])
        return self.update(out)

    def message(self, x_j):
        return x_j
    
    def aggregate(self, x, index):
        return scatter(x, index, 0, 2708)

    def update(self, inputs):
        return inputs

# (nodes, features)
node_feature = torch.randn(2708, 1433)
edges = torch.randint(0, 2708, (2, 10556))

gcn = GCN(1433, 16, 7)
gcn(node_feature, edges)