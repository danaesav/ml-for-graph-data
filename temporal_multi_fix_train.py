import numpy as np
import torch
from torch import optim, nn
from torch_geometric_temporal import temporal_signal_split

from dataset_loader import DatasetLoader
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig
from models.TemporalMultiFix import TemporalMultiFix

NUM_NODES = 10          # Must match N in generator config
NUM_FEATURES = 2        # m_rel = 2, total features = m_rel + m_irr + m_red = 2
NUM_LABELS = 5          # q = number of hyperspheres
NUM_TIMESTEPS = 15      # horizon
HIDDEN_DIM = 64
EPOCHS = 200
LR = 1e-2
THRESHOLD = 0.5         # for classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(config, train_ratio=0.8):
    dataset = DatasetLoader(config).get_dataset()
    return temporal_signal_split(dataset, train_ratio=train_ratio)


def initialize_model():
    deepwalk_embeddings = None #torch.randn(NUM_NODES, NUM_FEATURES)
    model = TemporalMultiFix(
        input_dim=NUM_FEATURES,
        num_of_nodes=NUM_NODES,
        output_dim=NUM_LABELS, dw_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, deepwalk_embeddings, optimizer, loss_fn


def train(model, train_dataset, deepwalk_embeddings, optimizer, loss_fn, epochs=EPOCHS):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # DynamicGraphTemporalSignal is iterable over time snapshots
        # It returns (x, edge_index, edge_weight, y) for each t
        for time, snapshot in enumerate(train_dataset):
            deep_walk_emb = None  #deepwalk_embeddings[time]
            optimizer.zero_grad()

            output = model(snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, deep_walk_emb)

            loss = loss_fn(output, snapshot.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / train_dataset.snapshot_count
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    config = TemporalMultiLabelGeneratorConfig(m_rel=2, m_irr=0, m_red=0, q=NUM_LABELS, N=NUM_NODES,
                                               max_r=0.7, min_r=0.1, mu=0, b=0.1, alpha=16, theta=np.pi / 7,
                                               horizon=NUM_TIMESTEPS)
    train_dataset, test_dataset = load_data(config)
    model, deepwalk_embeddings, optimizer, loss_fn = initialize_model()

    train(model, train_dataset, deepwalk_embeddings, optimizer, loss_fn)
    # TODO: evaluate(model, test_dataset, loss_fn)
