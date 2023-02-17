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
        self.prop1 = AGNNConv(requires_grad=False).jittable()
        self.prop2 = AGNNConv(requires_grad=True).jittable()
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

# try:
model = torch.jit.script(model)
# except Exception as e:
#     import logging
#     import os
#     result_file_root = "/workspace2/Project/gnn-perf-analysis/data/result/profile_jittable/"
#     log_path = os.path.join(result_file_root, "jittable_result" + ".log")
#     basic_file = os.path.basename(__file__)
#     logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode = 'a', format='%(levelname)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
#     logging.info("[script fail], "  + basic_file + ", " + repr(e))
#     exit()

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

import os
warm_up_epoch = os.getenv("WARM_UP_EPOCH")
if(warm_up_epoch):
    warm_up_epoch = eval(warm_up_epoch)
else:
    warm_up_epoch = 1

import ctypes
import time
import os

gpu_profile = os.getenv("FTXJ_GPU_PROF")
if(gpu_profile):
    gpu_profile = eval(gpu_profile)
else:
    gpu_profile = False
_cudart = ctypes.CDLL('libcudart.so')    

for epoch in range(1, warm_up_epoch + 2):    

    if(epoch > warm_up_epoch):
        start = time.time()
        if(gpu_profile):
            ret = _cudart.cudaProfilerStart()
            torch.cuda.nvtx.range_push("Profile") 


    train()
    (train_acc, val_acc, tmp_test_acc) = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')

if(gpu_profile):
    torch.cuda.nvtx.range_pop()
    ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
end = time.time()
print((end - start) * 1000 / 1)

exit()
