import torch
import torch.nn as nn
from torch import optim
from torch_geometric_temporal import temporal_signal_split
import numpy as np

from sklearn.metrics import f1_score
from GeneratorDatasetLoader import GeneratorDatasetLoader
from MultiLabelEvolveGCN import MultiLabelEvolveGCN
from temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig

# ----------- PARAMS -------------
NUM_NODES = 15              # Must match N in generator config
NUM_FEATURES = 2            # m_rel = 2, total features = m_rel + m_irr + m_red = 2
NUM_LABELS = 5              # q = number of hyperspheres
NUM_TIMESTEPS = 15          # horizon
HIDDEN_DIM = 64
EPOCHS = 30
LR = 0.01

config = TemporalMultiLabelGeneratorConfig(m_rel=2, m_irr=0, m_red=0, q=NUM_LABELS, N=NUM_NODES,
                                            max_r=0.7, min_r=0.1, mu=0, b=0.1,  alpha=16, theta=np.pi / 7,
                                            horizon=NUM_TIMESTEPS)
dataset = GeneratorDatasetLoader(config).get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

# ------------------------------------------------
# INSTANTIATE MODEL, LOSS, AND OPTIMIZER
# ------------------------------------------------

model = MultiLabelEvolveGCN(num_nodes=NUM_NODES, node_features=NUM_FEATURES, num_labels=NUM_LABELS)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.BCEWithLogitsLoss()

# ---------------------
# TRAINING LOOP
# ---------------------

model.train()
for epoch in range(200):
    total_loss = 0
    for snapshot in train_dataset:
        optimizer.zero_grad()
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / train_dataset.snapshot_count
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")


# ---------------------
# EVALUATION
# ---------------------
model.eval()
total_loss = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for snapshot in test_dataset:
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss_fn(y_hat, snapshot.y)
        total_loss += loss.item()

        # Apply sigmoid and threshold to get predictions
        probs = torch.sigmoid(y_hat)
        preds = (probs > 0.5).float() #the threshold ig you can play with it

        all_preds.append(preds.cpu())
        all_targets.append(snapshot.y.cpu())

avg_loss = total_loss / test_dataset.snapshot_count

# Concatenate all batches
all_preds = torch.cat(all_preds, dim=0).numpy()
all_targets = torch.cat(all_targets, dim=0).numpy()

val_f1 = f1_score(all_targets, all_preds, average='macro') # macro means compute per label and then average
print(f"Test BCE Loss: {avg_loss:.4f} | Validation Macro F1 Score: {val_f1:.4f}")
