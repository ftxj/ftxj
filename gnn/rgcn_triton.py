import argparse
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Entities
from torch_geometric.nn import FastRGCNConv, RGCNConv
from torch_geometric.utils import k_hop_subgraph

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

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AIFB',
                    choices=['AIFB', 'MUTAG', 'BGS', 'AM'])
args = parser.parse_args()

# Trade memory consumption for faster computation.
if args.dataset in ['AIFB', 'MUTAG']:
    RGCNConv = FastRGCNConv

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
dataset = Entities(path, args.dataset)
data = dataset[0]

# BGS and AM graphs are too big to process them in a full-batch fashion.
# Since our model does only make use of a rather small receptive field, we
# filter the graph to only contain the nodes that are at most 2-hop neighbors
# away from any training/test node.
node_idx = torch.cat([data.train_idx, data.test_idx], dim=0)
node_idx, edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx, 2, data.edge_index, relabel_nodes=True)

data.num_nodes = node_idx.size(0)
data.edge_index = edge_index
data.edge_type = data.edge_type[edge_mask]
data.train_idx = mapping[:data.train_idx.size(0)]
data.test_idx = mapping[data.train_idx.size(0):]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(data.num_nodes, 16, dataset.num_relations,
                              num_bases=30)
        self.conv2 = RGCNConv(16, dataset.num_classes, dataset.num_relations,
                              num_bases=30)

    def forward(self, edge_index, edge_type):
        
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu') if args.dataset == 'AM' else device
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

model = torch.compile(model)

def train():
    model.train()

    torch.cuda.nvtx.range_push("0-zero-grad")
    optimizer.zero_grad()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("1-forward")
    out = model(data.edge_index, data.edge_type)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2-loss")
    loss = F.nll_loss(out[data.train_idx], data.train_y)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("3-backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.nvtx.range_push("4-upgrad")
    optimizer.step()
    torch.cuda.nvtx.range_pop()

    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
    train_acc = float((pred[data.train_idx] == data.train_y).float().mean())
    test_acc = float((pred[data.test_idx] == data.test_y).float().mean())
    return train_acc, test_acc


for epoch in range(1, 51):

    torch.cuda.nvtx.range_push("Train-" + str(epoch))
    loss = train()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Test-" + str(epoch))
    train_acc, test_acc = test()
    torch.cuda.nvtx.range_pop()

    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()