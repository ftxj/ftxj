import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv

dataset = 'Cora'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]


import time
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()
torch._C._jit_set_nvfuser_enabled(True)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2).jittable()
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2).jittable()

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
# model = torch.jit.script(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x, data.edge_index, data.edge_attr)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


wramup = 10
test_epoc = 200
with torch.jit.fuser("fuser2"):
    for epoch in range(1, wramup + test_epoc):
        if(epoch == wramup):
            start = time.time()
    train()
    train_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
stop = time.time()

print("total =", stop-start)
print("each =", (stop-start) / test_epoc)