import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNH


# Original name: FPLPGCN_dw_linear
class TemporalMultiFix(nn.Module):
    def __init__(self, input_dim, num_of_nodes, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=None):
        super(TemporalMultiFix, self).__init__()

        # Layers for feature aggregation
        self.feature_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.feature_layers.append(EvolveGCNH(num_of_nodes=num_of_nodes, in_channels=input_dim))

        # Layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(EvolveGCNH(num_of_nodes=num_of_nodes, in_channels=output_dim))

        # TODO: self.fusion_layer = nn.Linear(input_dim + output_dim + dw_dim, output_dim)
        self.use_embedding = dw_dim is not None
        if self.use_embedding: # use embeddings
            self.fusion_layer = nn.Linear(input_dim + output_dim + dw_dim, output_dim)
        else: # no embeddings
            self.fusion_layer = nn.Linear(input_dim + output_dim, output_dim)

        self.FP = None
        self.LP = None
        self.dw_emb = None
        self.all_emb = None

    def forward(self, x, y, edge_index, edge_weight, deep_walk_emb=None):
        """
        x should be [num nodes, num features]
        edge index should be [num features, num edges]
        edge weight should be [num edges]
        """
        # feature_embeddings = []
        # label_embeddings = []

        # Feature propagation with evolving GCN
        for layer in self.feature_layers:
            x = layer(x, edge_index, edge_weight)
        FP = x
        # feature_embeddings.append(x)

        # Label propagation with evolving GCN
        x_label = y
        for layer in self.label_layers:
            x_label = layer(x_label, edge_index, edge_weight)
        LP = x_label
        # label_embeddings.append(x_label)

        # Stack along time dimension
        # feature_embeddings = torch.stack(feature_embeddings, dim=0)
        # label_embeddings = torch.stack(label_embeddings, dim=0)
        # deep_walk_emb_seq = torch.stack(deep_walk_emb_seq, dim=0)

        # For simplicity, use last time step embeddings
        # final_feature_emb = feature_embeddings[-1]
        # final_label_emb = label_embeddings[-1]
        # TODO: final_dw_emb = deep_walk_emb_seq[-1]

        if self.use_embedding and deep_walk_emb is not None:
            # TODO: x_fused = torch.cat((final_xfeature_emb, final_label_emb, final_dw_emb), dim=1)
            # print(final_feature_emb.shape, final_label_emb.shape, deep_walk_emb.shape)
            x_fused = torch.cat((FP, LP, deep_walk_emb), dim=1)
        else:
            x_fused = torch.cat([FP, LP], dim=-1)

        output = self.fusion_layer(x_fused)

        # in the original implementation they pass output through sigmoid, however here we will use BCEWithLogisLoss so we do not apply sigmoid here
        return output