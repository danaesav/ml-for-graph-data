from args import get_args
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv

from earlystopping import EarlyStopping
from data_loader import load_cora, load_citeseer

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
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
args.data_name = "citeseer"
args.hidden = 256
args.learning_rate = 0.01

if args.data_name == "cora":
    G = load_cora()

if args.data_name == "citeseer":
    G = load_citeseer()


model = SAGE(G.x.shape[1], args.hidden, G.num_class)


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












