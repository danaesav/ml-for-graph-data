from args import get_args
import os.path as osp
import time

import torch
import torch.nn.functional as F
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from earlystopping import EarlyStopping
from data_loader import load_cora, load_citeseer

from torch_sparse import SparseTensor
from torch_sparse import sum as sparsesum
from torch_sparse import spmm as sparsespmm

class GCN_LPA(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_gcn, lpa_iter):
        super().__init__()

        self.num_gcn = num_gcn
        # default: add self loop, normalize
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_gcn):
            if i == 0:
                self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
            elif i == self.num_gcn-1:
                self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))

        self.lpa_iter = lpa_iter
        self.edge_attr = torch.nn.Parameter(G.edge_weights.abs(), requires_grad=True)

    def forward(self, x, soft_labels, edge_index):
        weighted_adj = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=self.edge_attr, sparse_sizes=(x.shape[0], x.shape[0]))
        weighted_adj = row_normlize_sparsetensor(weighted_adj)
        #gcn

        x = F.dropout(x, p=0.5, training=self.training)
        #print("x", x)
        #print(self.edge_attr)

        x = F.relu(self.convs[0](x, weighted_adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, weighted_adj)

        #lpa
        predicted_labels = soft_labels

        #print("edge_index", edge_index)
        #print("edge_attr", self.edge_attr)
        #print("weighted adj", weighted_adj)
        _, _, value = weighted_adj.coo()
        for i in range(self.lpa_iter):
            predicted_labels = sparsespmm(edge_index, value, predicted_labels.shape[0], predicted_labels.shape[0], predicted_labels)
 
        return x, predicted_labels


def glorot(shape):
    init_range = np.sqrt(6.0 / np.sum(shape))
    #(r1 - r2) * torch.rand(a, b) + r2
    initial = (init_range-init_range) * torch.rand(shape) + (-init_range)

    return initial

def row_normlize_sparsetensor(a):

    deg = sparsesum(a, dim=1)
    #deg = a.sum(dim=1).to(torch.float)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0.)
    #deg_inv[deg_inv == float('inf')] = 0
    a_n = deg_inv.view(-1, 1) * a
    return a_n


def model_train():

    model.train()
    optimizer.zero_grad()

    out, predicted_labels = model(G.x, G.y_pad, G.edge_index)

    # loss calculation use logits and uncoded labels
    loss1 = F.cross_entropy(out[G.train_mask], G.uncode_label[G.train_mask])
    loss2 = F.cross_entropy(predicted_labels[G.train_mask], G.uncode_label[G.train_mask])

    loss = loss1+loss2

    loss.backward()
    optimizer.step()
    return float(loss)
    


@torch.no_grad()
def model_test():

    model.eval()
    out, predicted_labels = model(G.x, G.y_pad, G.edge_index)
    # validation loss for early stopping
    # loss calculation use logits and uncoded labels
    loss1 = F.cross_entropy(out[G.val_mask], G.uncode_label[G.val_mask])
    loss2 = F.cross_entropy(predicted_labels[G.val_mask], G.uncode_label[G.val_mask])
    loss_val = loss1 + loss2
    
    pred = out.argmax(dim=-1)
    accs = []
    for mask in [G.train_mask, G.val_mask, G.test_mask]:
        accs.append(int((pred[mask] == G.uncode_label[mask]).sum()) / int(mask.sum()))
    return accs, loss_val

args = get_args()
args.data_name = "citeseer"
args.hidden = 32
args.learning_rate = 0.2

if args.data_name == "cora":
    G = load_cora()

if args.data_name == "citeseer":
    G = load_citeseer()

print("######check if the labels are padded: ", G.y_pad[G.val_mask], G.y_pad[G.test_mask])
edge_weights = glorot(shape=G.edge_index.shape[1])
G.edge_weights = edge_weights

model = GCN_LPA(in_channels=G.x.shape[1], hidden_channels=args.hidden, out_channels=G.num_class,
                num_gcn=2, lpa_iter=5)


optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.learning_rate,
                             weight_decay=args.weight_decay
                            )

early_stopping = EarlyStopping(patience=args.patience, verbose=True)


for epoch in range(1, args.epochs):
    loss = model_train()
    results, loss_val = model_test()
    train_acc, val_acc, tmp_test_acc = results
    print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, '
          f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f},   Test acc: {tmp_test_acc:.4f}')
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

print("Optimization Finished!")
# load the last checkpoint with the best model
model.load_state_dict(torch.load('____checkpoint.pt'))
results, loss_val = model_test()
train_acc, val_acc, tmp_test_acc = results
print(f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f},   Test acc: {tmp_test_acc:.4f}'
      )












