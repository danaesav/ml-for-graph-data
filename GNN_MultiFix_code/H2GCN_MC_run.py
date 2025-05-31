from args import get_args
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from earlystopping import EarlyStopping
from data_loader import load_cora, load_citeseer

class H2GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(H2GCN, self).__init__()
        # input
        self.dense1 = torch.nn.Linear(nfeat, nhid)
        # output
        self.dense2 = torch.nn.Linear(nhid*7, nclass)
        # drpout
        # self.dropout = SparseDropout(dropout)
        self.dropout = dropout
        # conv
        self.conv1 = GCNConv(nhid, nhid)
        self.conv2 = GCNConv(nhid*2, nhid*2)
        self.relu = torch.nn.ReLU()
        self.vec = torch.nn.Flatten()
        self.iden = torch.sparse.Tensor()

    def forward(self, features, edge_index):

        # feature space ----> hidden
        # adj2 = adj * adj
        # r1: compressed feature matrix
        x = self.relu(self.dense1(features))
        # # vectorize
        # x = self.vec(x)
        # aggregate info from 1 hop away neighbor
        # r2 torch.cat(x, self.conv(x, adj), self.conv(x, adj2))
        x11 = self.conv1(x, edge_index)
        x12 = self.conv1(x11, edge_index)
        x1 = torch.cat((x11, x12), -1)

        # vectorize
        # x = self.vec(x1)
        # aggregate info from 2 hp away neighbor
        x21 = self.conv2(x1, edge_index)
        x22 = self.conv2(x21, edge_index)
        x2 = torch.cat((x21, x22), -1)

        # concat
        x = torch.cat((x, x1, x2), dim=-1)
        # x = self.dropout(x)
        x = F.dropout(x, self.dropout)
        x = self.dense2(x)

        return x


def model_train():

    model.train()
    optimizer.zero_grad()

    out = model(G.x, G.edge_index)

    # loss calculation use logits and uncoded labels
    loss = F.cross_entropy(out[G.train_mask], G.uncode_label[G.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)
    


@torch.no_grad()
def model_test():

    model.eval()
    out = model(G.x, G.edge_index)
    # validation loss for early stopping
    loss_val = F.cross_entropy(out[G.val_mask], G.uncode_label[G.val_mask])
    
    pred = out.argmax(dim=-1)
    accs = []
    for mask in [G.train_mask, G.val_mask, G.test_mask]:
        accs.append(int((pred[mask] == G.uncode_label[mask]).sum()) / int(mask.sum()))
    return accs, loss_val

args = get_args()
args.data_name = "cora"
args.hidden = 64
args.learning_rate = 0.001

if args.data_name == "cora":
    G = load_cora()

if args.data_name == "citeseer":
    G = load_citeseer()


model = H2GCN(nfeat=G.x.shape[1], nhid=args.hidden, nclass=G.num_class)


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












