import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

import time
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()
torch._C._jit_set_nvfuser_enabled(True)

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16).jittable()
        self.conv2 = GCNConv(16, dataset.num_classes).jittable()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
model = torch.compile(model)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

wramup = 10
test_epoc = 1000
# with torch.jit.fuser("fuser2"):
for epoch in range(1, wramup + test_epoc):
    if(epoch == wramup):
        start = time.time()
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
stop = time.time()

print("total =", stop-start)
print("each =", (stop-start) / test_epoc)

# Edge explainability
# ===================

# Captum assumes that for all given input tensors, dimension 0 is
# equal to the number of samples. Therefore, we use unsqueeze(0).
# captum_model = to_captum_model(model, mask_type='edge', output_idx=output_idx)
# edge_mask = torch.ones(data.num_edges, requires_grad=True, device=device)

# ig = IntegratedGradients(captum_model)
# ig_attr_edge = ig.attribute(edge_mask.unsqueeze(0), target=target,
#                             additional_forward_args=(data.x, data.edge_index),
#                             internal_batch_size=1)

# # Scale attributions to [0, 1]:
# ig_attr_edge = ig_attr_edge.squeeze(0).abs()
# ig_attr_edge /= ig_attr_edge.max()

# Visualize absolute values of attributions:
# explainer = Explainer(model)
# ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge)
# plt.show()

# # Node explainability
# # ===================

# captum_model = to_captum_model(model, mask_type='node', output_idx=output_idx)

# ig = IntegratedGradients(captum_model)
# ig_attr_node = ig.attribute(data.x.unsqueeze(0), target=target,
#                             additional_forward_args=(data.edge_index),
#                             internal_batch_size=1)

# # Scale attributions to [0, 1]:
# ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
# ig_attr_node /= ig_attr_node.max()

# # Visualize absolute values of attributions:
# ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge,
#                                      node_alpha=ig_attr_node)
# plt.show()

# # Node and edge explainability
# # ============================

# captum_model = to_captum_model(model, mask_type='node_and_edge',
#                                output_idx=output_idx)

# ig = IntegratedGradients(captum_model)
# ig_attr_node, ig_attr_edge = ig.attribute(
#     (data.x.unsqueeze(0), edge_mask.unsqueeze(0)), target=target,
#     additional_forward_args=(data.edge_index), internal_batch_size=1)

# # Scale attributions to [0, 1]:
# ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
# ig_attr_node /= ig_attr_node.max()
# ig_attr_edge = ig_attr_edge.squeeze(0).abs()
# ig_attr_edge /= ig_attr_edge.max()

# # Visualize absolute values of attributions:
# ax, G = explainer.visualize_subgraph(output_idx, data.edge_index, ig_attr_edge,
#                                      node_alpha=ig_attr_node)
# plt.show()
