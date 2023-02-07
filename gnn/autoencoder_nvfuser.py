import argparse
import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GAE, VGAE, GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=transform)
train_data, val_data, test_data = dataset[0]

import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()
torch._C._jit_set_nvfuser_enabled(True)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels).jittable()
        self.conv2 = GCNConv(2 * out_channels, out_channels).jittable()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels).jittable()
        self.conv_mu = GCNConv(2 * out_channels, out_channels).jittable()
        self.conv_logstd = GCNConv(2 * out_channels, out_channels).jittable()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels).jittable()

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_mu = GCNConv(in_channels, out_channels).jittable()
        self.conv_logstd = GCNConv(in_channels, out_channels).jittable()

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


in_channels, out_channels = dataset.num_features, 16

if not args.variational and not args.linear:
    model = GAE(GCNEncoder(in_channels, out_channels)).jittable()
elif not args.variational and args.linear:
    model = GAE(LinearEncoder(in_channels, out_channels))
elif args.variational and not args.linear:
    model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
elif args.variational and args.linear:
    model = VGAE(VariationalLinearEncoder(in_channels, out_channels))

model = torch.jit.script(model)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    
    torch.cuda.nvtx.range_push("1-forward-encode")
    z = model.encode(train_data.x, train_data.edge_index)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2-loss")
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    if args.variational:
        loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    torch.cuda.nvtx.range_pop()

    
    torch.cuda.nvtx.range_push("3-backward")
    loss.backward()
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("4-upgrad")
    optimizer.step()
    torch.cuda.nvtx.range_pop()

    x = float(loss)
    return x


@torch.no_grad()
def test(data):
    model.eval()
        
    torch.cuda.nvtx.range_push("1-forward-encode")
    z = model.encode(data.x, data.edge_index)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("2-forward-test")
    x = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
    torch.cuda.nvtx.range_pop()
    return x


with torch.jit.fuser("fuser2"):
    for epoch in range(1, args.epochs + 1):
        torch.cuda.nvtx.range_push("Train-" + str(epoch))
        loss = train()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("Test-" + str(epoch))
        y, pred = test(test_data)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        auc, ap = roc_auc_score(y, pred), average_precision_score(y, pred)
        torch.cuda.nvtx.range_pop()


        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')


ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()