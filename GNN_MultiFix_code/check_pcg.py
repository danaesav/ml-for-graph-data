import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import scipy
import scipy.io
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import Data
import torch_geometric.transforms as T
import argparse
import networkx as nx
import pandas as pd

def load_pcg(data_name, path=""):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()
    

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()


    feat = torch.tensor(pd.read_csv("pcg_removed_isolated_nodes/features.csv").to_numpy(), dtype=torch.float)
    print(feat.shape)
   

    #num_class = labels.shape[1]
    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    return G

G = load_pcg("pcg_removed_isolated_nodes")
print("shape of feature matrix", G.x.shape)
print("shape of label matrix: ", G.y.shape)
print(G.x[0])
print(G.x[-1])
# Assuming edge_index is a 2D array or list
flattened_edge_index = np.array(G.edge_index).flatten()
max_index = np.max(flattened_edge_index)

print("The maximum index in edge_index is:", max_index)
