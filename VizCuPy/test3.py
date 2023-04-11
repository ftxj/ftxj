"""
[Graph Attention Networks]
(https://arxiv.org/abs/1710.10903)
"""

import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset, FlickrDataset, RedditDataset
from torch.optim import Adam
import ctypes
import python_ops


class GATConv(nn.Module):
    def __init__(self, in_size, out_size, num_heads, dropout):
        super().__init__()

        self.out_size = out_size
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout)
        self.W = nn.Linear(in_size, out_size * num_heads)
        self.a_l = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.a_r = nn.Parameter(torch.zeros(1, out_size, num_heads))
        self.reset_parameters()
        self.softmax = dglsp.softmax

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.W.weight, gain=gain)
        nn.init.xavier_normal_(self.a_l, gain=gain)
        nn.init.xavier_normal_(self.a_r, gain=gain)

    ###########################################################################
    # (HIGHLIGHT) Take the advantage of DGL sparse APIs to implement
    # multihead attention.
    ###########################################################################
    def foo2(self, A):
        return A.row, A.col
    
    def foo3(self, A):
        return A.val

    def forward(self, A_hat, Z):
        
        row, col = self.foo2(A_hat)

        Z = self.dropout(Z)
        Z = self.W(Z).view(Z.shape[0], self.out_size, self.num_heads)

        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        e_l = (Z * self.a_l).sum(dim=1)
        e_r = (Z * self.a_r).sum(dim=1)

        tmp1 = e_l.index_select(0, row)
        tmp2 = e_r.index_select(0, col)
        
        e = tmp1 + tmp2

        a = F.leaky_relu(e)
        tmp = dglsp.val_like(A_hat, a)
        A_atten = self.softmax(tmp)

        a_drop = self.dropout(self.foo3(A_atten))
        A_atten = dglsp.val_like(A_atten, a_drop)
        return dglsp.bspmm(A_atten, Z)
    
    def jittable(self):
        self.softmax = python_ops.py_softmax
        return self

class GAT(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size=8, num_heads=8, dropout=0.6
    ):
        super().__init__()

        self.in_conv = GATConv(
            in_size, hidden_size, num_heads=num_heads, dropout=dropout
        )
        self.out_conv = GATConv(
            hidden_size * num_heads, out_size, num_heads=1, dropout=dropout
        )

    def forward(self, A, X):
        # Flatten the head and feature dimension.
        Z = F.elu(self.in_conv(A, X)).flatten(1)
        # Average over the head dimension.
        Z = self.out_conv(A, Z).mean(-1)
        return Z


def evaluate(g, pred):
    label = g.ndata["label"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # Compute accuracy on validation/test set.
    val_acc = (pred[val_mask] == label[val_mask]).float().mean()
    test_acc = (pred[test_mask] == label[test_mask]).float().mean()
    return val_acc, test_acc


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = FlickrDataset()
g = dataset[0].to(dev)

indices = torch.stack(g.edges())
N = g.num_nodes()
A = dglsp.spmatrix(indices, shape=(N, N))

I = dglsp.identity(A.shape, device=dev)
A_hat = A + I

X = g.ndata["feat"]
in_size = X.shape[1]
out_size = dataset.num_classes
model = GAT(in_size, out_size).to(dev)


import copy
def jittable(model):
    return_model = copy.deepcopy(model)
    if hasattr(return_model, 'jittable'):
        return return_model.jittable()
    for k, v in return_model._modules.items():
        if "Jittable" in v.__class__.__name__:
            assert 0, "Auto Profiler requires Layer not jittabled"
        if hasattr(v, 'jittable'):
            setattr(return_model, k, v.jittable())
    return return_model

model = jittable(model)



import ctypes

import ftxj.profiler as profiler
tracer = profiler.Tracer()

def train():
    global model
    global X
    A = A_hat.to('cuda')
    X = X.to('cuda')
    label = g.ndata["label"].to('cuda')
    train_mask = g.ndata["train_mask"].to('cuda')
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    for epoch in range(30):
        # Forward.
        # if epoch == 1:
        #     tracer.start()
        # if epoch == 2:
        #     model = torch.compile(model)
        # if epoch == 4:
        #     tracer.start()
        # if epoch == 20:
        #     tracer.start()

        model.train()

        # torch.cuda.nvtx.range_push(str(epoch) + "-E2E")

        # torch.cuda.nvtx.range_push(str(epoch) + "-forward")
        logits = model(A, X)
        # torch.cuda.nvtx.range_pop()



        # Compute loss with nodes in training set.
        loss = F.cross_entropy(logits[train_mask], label[train_mask])

        # Backward.
        optimizer.zero_grad()
        # torch.cuda.nvtx.range_push(str(epoch) + "-backward")
        loss.backward()
        # torch.cuda.nvtx.range_pop()

        # torch.cuda.nvtx.range_push(str(epoch) + "-step")
        optimizer.step()
        # torch.cuda.nvtx.range_pop()
        # Compute prediction.
        model.eval()
        # torch.cuda.nvtx.range_push(str(epoch) + "-eval")
        logits = model(A_hat, X)
        pred = logits.argmax(dim=1)
        # Evaluate the prediction.
        val_acc, test_acc = evaluate(g, pred)
        # torch.cuda.nvtx.range_pop()

        # torch.cuda.nvtx.range_pop()



        # if epoch == 1:
        #     tracer.stop()
        # if epoch == 4:
        #     tracer.stop()
        # if epoch == 20:
        #     tracer.stop()


        print(
            f"In epoch {epoch}, loss: {loss:.3f}, val acc: {val_acc:.3f}, test"
            f" acc: {test_acc:.3f}"
        )




train()


# tracer.print()

# tracer.save()

def to_json(x):
    data = x.dump()
    k = []
    for l in data:
        new_l = []
        for x in l:
            if "timeline_split" not in x['name']:
                new_l.append(x)
        k = k + new_l
    import json
    data2 = {}
    data2['traceEvents'] = k
    
    data2 = json.dumps(data2, indent=4)
    with open("sample7.json", "w") as outfile:
        outfile.write(data2)



to_json(tracer)