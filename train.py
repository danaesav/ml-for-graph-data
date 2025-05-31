import torch.nn as nn
from torch import optim
from torch_geometric_temporal import temporal_signal_split
import numpy as np
from tqdm import tqdm

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
for epoch in tqdm(range(200)):
    total_loss = 0
    for snapshot in train_dataset:
        optimizer.zero_grad()
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = loss_fn(y_hat, snapshot.y)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
    # avg_loss = total_loss / (time + 1) ?
    # if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")
