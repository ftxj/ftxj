import torch
from torch import Tensor

def index_select_elementwise_reduce(x: Tensor, row, col):
    x_i = torch.index_select(x, 0, row)
    x_j = torch.index_select(x, 0, col)
    return torch.sum(x_i + x_j, -1)


graph_data = [
    [100, 200, 4],
    [10_000, 200_000, 64],
    [100_000, 2_000_000, 32]
]

num_nodes, num_edges, feature = graph_data[2]
x = torch.randn(num_nodes, feature, device="cuda")
edge_index = torch.randint(num_nodes, (2, num_edges), device="cuda")
row, col = edge_index


fn = torch.compile(index_select_elementwise_reduce)    

for i in range(5):
    fn(x, row, col)