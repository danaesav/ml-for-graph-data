import warnings
warnings.filterwarnings("ignore")
import torch.optim as optim
import argparse
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
import os
import os.path as osp
from torch_geometric.loader import NeighborLoader
import copy
import numpy as np
from torch_geometric.utils import add_self_loops
import scipy
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def import_yelp(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='../data/Yelp')
    data = dataset[0]
    labels = data.y
    features = data.x

    edge_index = data.edge_index

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             y=labels,
             edge_index=edge_index)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.n_id = torch.arange(G.x.shape[0])

    return G


def load_hyper_data(data_name, edge_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            skip_header=1, dtype=np.dtype(float), delimiter=',')).float()

    # varying homophily with differet edges
    edges = torch.tensor(np.genfromtxt(os.path.join(path, edge_name),
                         dtype=np.dtype(float), delimiter=',')).long()

    edge_index = torch.transpose(edges, 0, 1)
    print(edge_index.shape)

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    y = labels.clone().detach().float()
    y[val_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])
    y[test_mask] = torch.full((1, labels.shape[1]), 1 / labels.shape[1])

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.soft_labels = y
    G.n_id = torch.arange(num_nodes)

    return G


def import_dblp(data_name, split_name, train_percent, path="../../../data/dblp"):
    print('Loading dataset ' + data_name + '...')
    features = torch.FloatTensor(np.genfromtxt(os.path.join(path, "features.txt"), delimiter=",", dtype=np.float64))
    labels = torch.FloatTensor(np.genfromtxt(os.path.join(path, "labels.txt"), delimiter=","))
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "dblp.edgelist"))).long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             y=labels,
             edge_index=edge_index)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.n_id = torch.arange(G.x.shape[0])

    return G


def load_mat_data(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()
    edge_index = torch.transpose(torch.nonzero(adj.to_dense()), 0, 1).long()
    # for gcnlpa
    #edge_index = add_self_loops(edge_index)[0]
    # prepare the feature matrix
    #features = torch.range(0, labels.shape[0] - 1).long()
    #features = torch.rand(labels.shape[0], 128)
    features = torch.eye(labels.shape[0])

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    y = labels.clone().detach().float()
    y[val_mask] = torch.zeros(labels.shape[1])
    y[test_mask] = torch.zeros(labels.shape[1])

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.n_id = torch.arange(num_nodes)

    return G


def import_ogb(data_name):
    print('Loading dataset ' + data_name + '...')

    dataset = PygNodePropPredDataset(name=data_name, transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    row, col, _ = data.adj_t.coo()
    edge_index = torch.vstack((row, col))

    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    val_mask[valid_idx] = True

    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    num_nodes = data.x.shape[0]

    G = Data(x=data.x,
             edge_index=edge_index,
             y=data.y.float())
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.n_id = torch.arange(num_nodes)

    # label embedding to start with
    lbl_emb = torch.arange(G.y.shape[1]).long()
    G.lbl_emb = lbl_emb
    G.n_id = torch.arange(num_nodes)

    return G


def load_pcg(data_name, split_name, train_percent, path="../../data/"):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    num_nodes = features.shape[0]
    G.n_id = torch.arange(num_nodes)

    return G


def load_humloc(data_name="HumanGo", path="../../data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                             dtype=np.dtype(float), delimiter=',')

    features = torch.tensor(features).float()

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    num_nodes = features.shape[0]
    G.n_id = torch.arange(num_nodes)

    return G


def load_eukloc(data_name="EukaryoteGo", path="../../data/"):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    num_nodes = features.shape[0]
    G.n_id = torch.arange(num_nodes)

    return G

