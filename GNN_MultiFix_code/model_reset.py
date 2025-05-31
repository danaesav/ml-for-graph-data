import torch.nn as nn
from torch_geometric.nn import GCNConv
from metrics import *
from torch.nn import functional as F
from torch_sparse import SparseTensor

class FPLPGCN_dw(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim + dw_dim, output_dim)
        self.fuse_FP_dw = nn.Linear(hidden_dim + dw_dim, output_dim)

        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

    def forward(self, x, y, edge_index, deep_walk_emb, label_input_mask):

        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)
            ####################### reset #################################
            x_label[label_input_mask] = y[label_input_mask]
            ##################################################################


        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        x_fused = self.fusion_layer(x_fused)
        #print("generated embeddings: ", x_fused)
        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused


class FPLPGCN_dw_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw_MLP, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Sequential(
                                          nn.Linear(hidden_dim + output_dim + dw_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, output_dim)
                                         )

        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

    def forward(self, x, y, edge_index, deep_walk_emb, label_input_mask):

        # GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
            if i != len(self.gcn_layers) - 1:
                x = F.relu(x)

        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            if j != len(self.label_layers) - 1:
                x_label = F.relu(x_label)
            ####################### reset #################################
            x_label[label_input_mask] = y[label_input_mask]
            ##################################################################


        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        x_fused = self.fusion_layer(x_fused)
        #print("generated embeddings: ", x_fused)

        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused
    



class FPLPGCN_dw_linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10, dw_dim=64, multi_class=False):
        super(FPLPGCN_dw_linear, self).__init__()

        # GCN layers for feature aggregation
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Linear layers for label aggregation
        self.label_layers = nn.ModuleList()
        for _ in range(num_label_layers):
            self.label_layers.append(GCNConv(output_dim, output_dim))

        # Linear layer for fusion
        self.fusion_layer = nn.Linear(hidden_dim + output_dim + dw_dim, output_dim)
        ################# for test only the label propagation #################
        #self.fusion_layer = nn.Linear(dw_dim, output_dim)
        #####################################################################################
        self.smd = torch.nn.Sigmoid()

        self.multi_class = multi_class

        self.FP = None
        self.LP = None
        self.dw_emb = None
        self.all_emb = None

    def forward(self, x, y, edge_index, deep_walk_emb, label_input_mask):

        #GCN layers for feature aggregation
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, edge_index)
        self.FP = x
        #print("FP: ", self.FP[:10])

        ####################################################################
        x_label = y
        for j, label_layer in enumerate(self.label_layers):
            x_label = label_layer(x_label, edge_index)
            ####################### reset #################################
            x_label[label_input_mask] = y[label_input_mask]
            ##################################################################
        self.LP = x_label
        # print("LP: ", self.LP)



        self.dw_emb = deep_walk_emb
        # print("dw emb", self.dw_emb)
        ####################################################################
        # Fusion of feature propagation and label propagation
        x_fused = torch.cat((x, x_label, deep_walk_emb), dim=1)
        #x_fused = torch.cat((x, x_label), dim=1)
        
        #x_fused = torch.cat((x, x_label), dim=1)
        #x_fused = x_label
        #x_fused = self.dw_emb

        ####################################################################
        self.all_emb = x_fused
        x_fused = self.fusion_layer(x_fused)
        #print("generated embeddings: ", x_fused[:10])

        if not self.multi_class:
            return self.smd(x_fused)
        else:
            return x_fused





