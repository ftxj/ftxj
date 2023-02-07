import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()
torch._C._jit_set_nvfuser_enabled(True)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6).jittable()
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6).jittable()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
model = torch.jit.script(model)

def train():
    model.train()

    torch.cuda.nvtx.range_push("0-zero-grad")
    optimizer.zero_grad()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("1-forward")
    out = model(data.x, data.edge_index)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2-loss")
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
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

    torch.cuda.nvtx.range_push("1-forward")
    pred = model(data.x, data.edge_index).argmax(dim=-1)
    torch.cuda.nvtx.range_pop()

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = final_test_acc = 0
with torch.jit.fuser("fuser2"):
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()