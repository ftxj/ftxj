import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ARMAConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


import time
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()

def act_fn(x):
    return x
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25).jittable()

        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=3,
                              num_layers=2, shared_weights=True, dropout=0.25,
                              act=act_fn).jittable()

    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net(dataset.num_features, 16,
                  dataset.num_classes).to(device), data.to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test():
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
wramup = 10
test_epoc = 100
for epoch in range(1, wramup + test_epoc):
    if(epoch == wramup):
        start = time.time()
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
stop = time.time()

print("total =", stop-start)
print("each =", (stop-start) / test_epoc)