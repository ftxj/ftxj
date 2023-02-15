import os.path as osp

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import (
    completeness_score,
    homogeneity_score,
    v_measure_score,
)
from torch.nn import Linear

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ARGVA, GCNConv
import time
import ctypes
_cudart = ctypes.CDLL('libcudart.so')
ret = _cudart.cudaProfilerStart()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, name='Cora', transform=transform)
train_data, val_data, test_data = dataset[0]


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)


encoder = Encoder(train_data.num_features, hidden_channels=32, out_channels=32)
encoder = torch.compile(encoder)
discriminator = Discriminator(in_channels=32, hidden_channels=64,
                              out_channels=32)
discriminator = torch.compile(discriminator)

model = ARGVA(encoder, discriminator).to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.005)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                           lr=0.001)


def train():
    model.train()
    encoder_optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)

    # We optimize the discriminator more frequently than the encoder.
    for i in range(5):
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        discriminator_loss.backward()
        discriminator_optimizer.step()

    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + model.reg_loss(z)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    encoder_optimizer.step()
    return float(loss)


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)

    # Cluster embedded values using k-means.
    kmeans_input = z.cpu().numpy()
    kmeans = KMeans(n_clusters=7, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    labels = data.y.cpu().numpy()
    completeness = completeness_score(labels, pred)
    hm = homogeneity_score(labels, pred)
    nmi = v_measure_score(labels, pred)

    auc, ap = model.test(z, data.pos_edge_label_index,
                         data.neg_edge_label_index)

    return auc, ap, completeness, hm, nmi

wramup = 10
test_epoc = 100
for epoch in range(1, wramup + test_epoc):
    if(epoch == wramup):
        start = time.time()
    loss = train()
    auc, ap, completeness, hm, nmi = test(test_data)
    print((f'Epoch: {epoch:03d}, Loss: {loss:.3f}, AUC: {auc:.3f}, '
           f'AP: {ap:.3f}, Completeness: {completeness:.3f}, '
           f'Homogeneity: {hm:.3f}, NMI: {nmi:.3f}'))



ret = _cudart.cudaProfilerStop()

torch.cuda.synchronize()
stop = time.time()

print("total =", stop-start)
print("each =", (stop-start) / test_epoc)