import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import AGNNConv

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False).jittable()
        self.prop2 = AGNNConv(requires_grad=True).jittable()
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    def forward(self, data_x, data_edge_index):
                
        torch.cuda.nvtx.range_push("0-dropout+lin1+relu")
        x = F.dropout(data_x, training=self.training)
        x = F.relu(self.lin1(x))
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("1-agnn")
        x = self.prop1(x, data_edge_index)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("2-agnn")
        x = self.prop2(x, data_edge_index)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("3-dropout+lin2")
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("4-log_softmax")
        x = F.log_softmax(x, dim=1)
        torch.cuda.nvtx.range_pop()

        return x

# torch._C._jit_set_nvfuser_enabled(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model = torch.jit.script(model)
def train():
    model.train()
    torch.cuda.nvtx.range_push("0-zero-grad")
    optimizer.zero_grad()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("1-forward")
    out = model(data.x, data.edge_index)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2-loss")
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("3-backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("4-upgrad")
    optimizer.step()
    torch.cuda.nvtx.range_pop()


@torch.no_grad()
def test():
    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
# with torch.jit.fuser("fuser2"):
for epoch in range(1, 201):
    torch.cuda.nvtx.range_push("Train-" + str(epoch))
    train()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("Test-" + str(epoch))
    train_acc, val_acc, tmp_test_acc = test()
    torch.cuda.nvtx.range_pop()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
        f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

 

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
