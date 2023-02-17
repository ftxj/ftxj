
import argparse
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
from torch_geometric.nn import AGNNConv
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    def forward(self):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.prop1(x, data.edge_index)
        x = self.prop2(x, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(model, data) = (Net().to(device), data.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def test():
    model.eval()
    (logits, accs) = (model(), [])
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
