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
torch._C._jit_set_nvfuser_enabled(True)



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
        self.shape_x = None
        self.index_size = None
    def forward(self, x, edge_index):

        if(self.index_size == None):
            self.index_size = edge_index.size()
        else:
            if(self.index_size != edge_index.size()):
                print("dynamic shape")


        if(self.shape_x == None):
            self.shape_x = [x.size()]
        else:
            if(self.shape_x[0] != x.size()):
                print("dynamic shape")

        x = F.dropout(x, training=self.training)
        if(len(self.shape_x) == 1):
            self.shape_x.append(x.size())
        else:
            if(self.shape_x[1] != x.size()):
                print("dynamic shape")

        x = F.relu(self.conv1(x, edge_index))
        if(len(self.shape_x) == 2):
            self.shape_x.append(x.size())
        else:
            if(self.shape_x[2] != x.size()):
                print("dynamic shape")


        x = F.dropout(x, training=self.training)
        if(len(self.shape_x) == 3):
            self.shape_x.append(x.size())
        else:
            if(self.shape_x[3] != x.size()):
                print("dynamic shape")

        x = self.conv2(x, edge_index)
        if(len(self.shape_x) == 4):
            self.shape_x.append(x.size())
        else:
            if(self.shape_x[4] != x.size()):
                print("dynamic shape")

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net(dataset.num_features, 16,
                  dataset.num_classes).to(device), data.to(device)
# model = torch.jit.script(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()

    torch.cuda.synchronize()
    start1 = time.time()

    out = model(data.x, data.edge_index)
    torch.cuda.synchronize()
    stop1 = time.time()

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])

    torch.cuda.synchronize()
    start2 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    stop2 = time.time()

    torch.cuda.synchronize()
    start3 = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    stop3 = time.time()

    return stop1 - start1, stop2 - start2, stop3 - start3

def test():
    model.eval()

    torch.cuda.synchronize()
    start = time.time()

    out, accs = model(data.x, data.edge_index), []
    
    torch.cuda.synchronize()
    stop = time.time()

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs, stop-start


best_val_acc = test_acc = 0
wramup = 10
test_epoc = 100

min_time = 100000000
forward_min_time = 100000000
train_forward_time = 100000000
train_backward_time = 100000000
train_upgrad_time = 100000000

with torch.jit.fuser("fuser2"):
    for epoch in range(1, wramup + test_epoc):
        
        torch.cuda.synchronize()
        start = time.time()

        t1, t2, t3 = train()
        
        acc, forward_time = test()

        torch.cuda.synchronize()
        stop = time.time()

        train_acc, val_acc, tmp_test_acc = acc

        min_time = min(min_time, stop - start)
        forward_min_time = min(forward_time, forward_min_time)
        train_forward_time = min(t1, train_forward_time)
        train_backward_time = min(t2, train_backward_time)
        train_upgrad_time = min(t3, train_upgrad_time)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
            f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
stop = time.time()

print("e2e_min = ", min_time)
print("forward = ", forward_min_time)
print("train_forward_time = ", train_forward_time)
print("train_backward_time = ", train_backward_time)
print("train_upgrad_time = ", train_upgrad_time)
# print("each =", (stop-start) / test_epoc)