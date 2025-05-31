from torch import nn
from torch_geometric_temporal import EvolveGCNH

class MultiLabelEvolveGCN(nn.Module):
    def __init__(self, num_nodes, node_features, num_labels):
        super().__init__()
        self.evolvegcn = EvolveGCNH(num_of_nodes=num_nodes, in_channels=node_features)
        self.linear = nn.Linear(node_features, num_labels)  # maps embedding -> multi-label outputs

    def forward(self, x, edge_index, edge_weight=None):
        h = self.evolvegcn(x, edge_index, edge_weight).relu()
        h = self.linear(h)  # Shape: [num_nodes, num_labels]
        return h