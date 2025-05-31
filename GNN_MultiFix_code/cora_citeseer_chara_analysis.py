from torch_geometric.datasets import Planetoid
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import jaccard_score
from torch.nn.functional import one_hot

def calculate_homophily(edge_index, labels):
    row, col = edge_index
    edge_labels = torch.stack((labels[row], labels[col]), dim=0)
    return (edge_labels[0] == edge_labels[1]).sum().item() / edge_labels.shape[1]

def my_homophily(edge_index, labels):
    edges = torch.transpose(edge_index, 0, 1)

    # one-hot encode labels
    one_hot_labels = one_hot(data.y)

    support = 0.0
    for edge in edges:
        support = support + jaccard_score(one_hot_labels[edge[0].item()], one_hot_labels[edge[1].item()])
    
    h = support / edges.shape[0]
    return h


datasets = ['Cora', 'CiteSeer']
for dataset_name in datasets:
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]

    # calculate homophily
    homophily = calculate_homophily(data.edge_index, data.y)
    my_homo = my_homophily(data.edge_index, data.y)
    print(f'Homophily in {dataset_name}: {homophily}')
    print(f'My version of Homophily in {dataset_name}: {my_homo}')

    # calculate label distribution
    G = to_networkx(data, to_undirected=True)
    avg_clustering_coefficient = nx.average_clustering(G)
    print(f'Average clustering coefficient in {dataset_name}: {avg_clustering_coefficient}')