import numpy as np
import os
import scipy
import scipy.io
import os.path as osp
from torch_geometric.data import Data
import torch
from torch_geometric.utils import degree, subgraph, to_networkx, to_undirected
import networkx as nx
import matplotlib.pyplot as plt
import random

def load_hyper_data(data_name="Hyperspheres_10_10_0", split_name="split_0.pt", train_percent=0.6, path=""):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = np.genfromtxt(osp.join(path, data_name, "features.csv"),
                             skip_header=1, dtype=np.dtype(float), delimiter=',')

    edges = torch.tensor(np.genfromtxt(osp.join(path, data_name, "edges.txt"),
                         dtype=np.dtype(float), delimiter=',')).long()
    edge_index = torch.transpose(edges, 0, 1)


    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    
    G.num_nodes = G.y.shape[0]

    return G


def get_direct_neighbors(node, edge_index):
    # Convert the graph to undirected
    undirected_edge_index = to_undirected(edge_index)

    # Get the immediate neighbors of the node
    immediate_neighbors = undirected_edge_index[1][undirected_edge_index[0] == node]

    return immediate_neighbors

def get_subgraph(nodes, edge_index, edge_attr=None):
    # Get the subgraph that includes only the given nodes
    new_edge_index, new_edge_attr = subgraph(nodes, edge_index, edge_attr=edge_attr, relabel_nodes=False)

    return new_edge_index, new_edge_attr


def visualize_subgraph(edge_index, label_vec):
    # Convert the edge index to a NetworkX graph
    # Create an empty NetworkX graph
    G = nx.Graph()

    # Add edges to the graph
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

    # Draw the graph
    nx.draw(G, with_labels=True)

    plt.savefig("visualize_subgraph")
    

G = load_hyper_data()

# calcuate median value of node degree
degrees = degree(G.edge_index[0]).long()

# Select a node with the median degree
nodes_with_6_neighbors = (degrees == 6).nonzero().squeeze().tolist()
#nodes_with_median_degree = (degrees == median_degree).nonzero(as_tuple=True)[0]
selected_node = nodes_with_6_neighbors
print("selected node is: ", selected_node)

# the direct neighbors
immediate_neighbors = get_direct_neighbors(selected_node, G.edge_index)
print("labels of the node", G.y[selected_node])

immediate_neighbors_labels = G.y[immediate_neighbors]
hop1_label_distribution = immediate_neighbors_labels.sum(dim=0)
print("the label distribution in direct neighborhood: ", hop1_label_distribution)


# Get the subgraph that includes only the given nodes
# add the node itself
immediate_neighbors_and_node = torch.cat((immediate_neighbors, torch.tensor([selected_node])))

for n in immediate_neighbors_and_node:
    print("node ", n, " with labels: ", G.y[n].nonzero().flatten())
    
new_edge_index, _ = get_subgraph(immediate_neighbors_and_node, G.edge_index, G.edge_attr)

# Visualize the subgraph
labels_nodes = G.y[immediate_neighbors_and_node]
visualize_subgraph(new_edge_index, labels_nodes)


