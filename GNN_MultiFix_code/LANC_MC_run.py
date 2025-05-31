import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Conv1d, MaxPool1d, Linear
import os
import torch
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp


import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from args import get_args
from torch_geometric.utils import degree
import networkx as nx
from torch_geometric.utils import to_networkx
import torch.optim as optim
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from earlystopping import EarlyStopping
from torch.utils.data import DataLoader
from metrics import *
from data_loader import load_cora, load_citeseer


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class LANC(torch.nn.Module):
    def __init__(self, in_channels, class_channels, num_label):
        super().__init__()
        self.conv1 = Conv1d(in_channels, 16, 2)
        self.conv2 = Conv1d(in_channels, 16, 3)
        self.conv3 = Conv1d(in_channels, 16, 4)
        self.conv4 = Conv1d(in_channels, 16, 5)
        self.mlp = nn.Sequential(
                                Linear(192, 64),
                                nn.ReLU(),
                                Linear(64, class_channels))
        self.mlp1 = nn.Sequential(Linear(128, 64),
                                  nn.ReLU(),
                                  Linear(64, class_channels))

        self.attention = nn.Sequential(nn.Linear(192, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 1, bias=False))
        self.lbl_emb = nn.Embedding(num_label, 128)

    def forward(self, x, y):
        y = self.lbl_emb(y)

        x = x.permute(0, 2, 1)
        #convolution
        x1 = self.conv1(x)

        x1 = F.relu(x1)
        #print(x1[0])
        x1 = F.dropout(x1, p=0.6)
        #print(x1[0])
        #x1 = F.dropout(F.relu(self.conv1(x)), p=0.6)

        #print(x1[0])
        #print("%%%%%%%%%%%%%%%%%%%%%%")
        x2 = F.dropout(F.relu(self.conv2(x)), p=0.6)
        x3 = F.dropout(F.relu(self.conv3(x)), p=0.6)
        x4 = F.dropout(F.relu(self.conv4(x)), p=0.6)
        #print("%%%%%%%%%%%%%%%%%%%%")
        #print(x1.shape)
        #max pooling
        x1 = torch.amax(x1, 2)
        x2 = torch.amax(x2, 2)
        x3 = torch.amax(x3, 2)
        x4 = torch.amax(x4, 2)
        # feature vector
        out = torch.cat((x1, x2, x3, x4), dim=1)
        # (64, 64)
        #print(out[0:3])
        # attention vector
        # (batch_size, node embedding + label embedding dimension)
        s = []
        for i in range(y.shape[0]):
            # 64 is the batch size
            support = torch.cat(out.shape[0] * [y[i]]).reshape(out.shape[0], -1)
            # concat the node emb with one label emb
            c = torch.hstack((out, support))
            s.append(c)
        #print(torch.stack(s).shape)
        s = self.attention(torch.stack(s))#.squeeze()
        #print(s.shape)
        s = s.squeeze()
        #print('s')
        #print(s.shape)
        #print(s[0:3])
        a = F.softmax(torch.transpose(s, 0, 1), dim=1)
        #print("a")
        #print(a[0:3])

        att_vec = torch.mm(a, y)

        # concat feature vector and attention vector
        emb = torch.cat((out, att_vec), dim=1)

        # use label embedding to predict the labels
        emb_pre = self.mlp(emb)
        #print('prediction from embedding')
        #print(emb_pre[0])
        #print(emb_pre.shape)
        # try padding in label embedding and use the same mlp as emb
        pad = torch.zeros(y.shape[0], emb.shape[1] - y.shape[1])
        y = torch.hstack((pad, y))
        y_pre = self.mlp(y)
        #############
        # y_pre = self.mlp1(y)
        #y_pre = self.mlp1(y)
        return emb_pre, y_pre


def model_train(train_loader):
    un_lbl = torch.arange(0, G.num_class)

    outs1 = []
    total_loss = 0
    for idx in train_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, out2 = model.forward(x, y)
        #print(out1.shape)
        outs1.append(out1)

        loss_train = F.cross_entropy(out1, G.uncode_label[idx]) + F.cross_entropy(out2, un_lbl)
        loss_train.backward()
        optimizer.step()

        total_loss += float(loss_train) * len(idx)

    output1 = torch.cat(outs1, dim=0)

    # pred are all from training data
    pred = output1.argmax(dim=-1)

    acc_train = int((pred == G.uncode_label[G.train_mask]).sum()) / int(G.train_mask.sum())

    return total_loss/G.num_nodes, acc_train


def model_test(data_loader):
    un_lbl = torch.arange(0, G.num_class)
    # all embedding output
    outs1 = []
    for idx in data_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, output2 = model.forward(x, y)
        outs1.append(out1)

    # embedding prediction
    output1 = torch.cat(outs1, dim=0)
    # calculate loss
   
    loss_val = F.cross_entropy(output1[G.val_mask], G.uncode_label[G.val_mask]) + F.cross_entropy(output2, un_lbl)

    pred = output1.argmax(dim=-1)
    accs = []
    for mask in [G.train_mask, G.val_mask, G.test_mask]:
        accs.append(int((pred[mask] == G.uncode_label[mask]).sum()) / int(mask.sum()))

    return loss_val, accs

if __name__ == "__main__":
    args = get_args()
    
    args.batch_sie = 64

    if args.data_name == "cora":
        G = load_cora()
    elif args.data_name == "citeseer":
        G = load_citeseer()

    G.n_id = torch.arange(G.num_nodes)

    # adj
    adj = torch.sparse_coo_tensor(G.edge_index, torch.ones(G.edge_index.shape[1]),
                                  (G.x.shape[0], G.x.shape[0]))
    adj_dense = adj.to_dense()
    G.adj = adj

    # label embeddings
    lbl_emb = torch.arange(G.num_class).long()
    G.lbl_emb = lbl_emb
    num_nodes = G.x.shape[0]

    # maximum degree
    degree = 0

    for row in adj_dense:
        deg = torch.flatten(torch.nonzero(row)).shape[0]
        if deg > degree:
            degree = deg

    # prepare the feature matrix
    attrs = []
    for i in range(G.num_nodes):
        neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
        if len(neigh_ind) < degree:
            num_to_pad = degree - len(neigh_ind)
            padding = torch.zeros(num_to_pad, G.x.shape[1])
            # featues of neighbors
            attr = G.x[neigh_ind]
            attr = torch.vstack((attr, padding))
        else:
            attr = G.x[neigh_ind]
            # featues of neighbors
        attrs.append(attr)

    G.x = torch.stack(attrs)


    train_loader = DataLoader(G.n_id[G.train_mask],
                              shuffle=False,
                              batch_size=64,
                              num_workers=0)
    eva_loader = DataLoader(G.n_id,
                            shuffle=False,
                            batch_size=64,
                            num_workers=0)

    model = LANC(in_channels=G.x.shape[2],
                 class_channels=G.num_class,
                 num_label=G.lbl_emb.shape[0])

    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    early_stopping = EarlyStopping(model_name=args.data_name,
                                   patience=args.patience, verbose=True)

    for epoch in range(1, args.epochs):

        loss_train, acc_train = model_train(train_loader)
        loss_val, results = model_test(eva_loader)
        train_acc, val_acc, tmp_test_acc = results

        print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
              f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, Test acc: {tmp_test_acc:.4f}')
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Optimization Finished!")
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(args.data_name + "___"+'_checkpoint.pt'))
    loss_val, results = model_test(eva_loader)
    train_acc, val_acc, tmp_test_acc = results
    print(f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f},   Test acc: {tmp_test_acc:.4f}'
        )



