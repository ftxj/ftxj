
import argparse
import time
parser = argparse.ArgumentParser(description=__file__ + " Testing")

parser.add_argument('--warmup', default=5, type=int, 
                    help='Number of warmup iteration')

parser.add_argument('--iter-time', default=1, type=int, 
                    help='Number of perf iteration')
parser.add_argument('--gpu-profile', default=False, type=bool, 
                    help='Using cuda profile')
args = parser.parse_args()


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

class Net(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.25)
        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.25, act=lambda x: x)

    @torch._dynamo.optimize("ts_nvfuser")
    def forward(self, x, edge_index):
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(model, data) = (Net(dataset.num_features, 16, dataset.num_classes).to(device), data.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def test():
    model.eval()
    (logits, accs) = (model(data.x, data.edge_index), [])
    for (_, mask) in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
best_val_acc = test_acc = 0

warm_up_epoch = args.warmup
perf_epoch = args.iter_time
for epoch in range(1, warm_up_epoch + perf_epoch):    

    if(epoch == warm_up_epoch):
        start = time.time()


    train()
    (train_acc, val_acc, tmp_test_acc) = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

torch.cuda.synchronize()
end = time.time()
print((end - start) * 1000 / 1)
exit()
