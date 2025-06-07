import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch import optim, nn
from torch_geometric_temporal import temporal_signal_split

from dataset_loader import DatasetLoader
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig
from models.TemporalMultiFix import TemporalMultiFix

NUM_NODES = 50          # Must match N in generator config
NUM_FEATURES = 10        # m_rel = 2, total features = m_rel + m_irr + m_red = 2
NUM_LABELS = 20          # q = number of hyperspheres
NUM_TIMESTEPS = 15      # horizon
HIDDEN_DIM = 64
EPOCHS = 100
LR = 1e-2
THRESHOLD = 0.5         # for classification
EMBEDDING = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def temporal_signal_split_list(data, train_ratio):
    train_snapshots = int(train_ratio * len(data))
    
    train_iterator = data[0:train_snapshots]
    test_iterator = data[train_snapshots:]

    return train_iterator, test_iterator


def load_data(config, train_ratio=0.8):
    dataset, embeddings = DatasetLoader(config).get_dataset()
    return *temporal_signal_split(dataset, train_ratio=train_ratio), *temporal_signal_split_list(embeddings, train_ratio=train_ratio)


def initialize_model():
    # deepwalk_embeddings = None #shape(TIME, NUM_NODES, NUM_FEATURES)
    model = TemporalMultiFix(
        input_dim=NUM_FEATURES,
        num_of_nodes=NUM_NODES,
        output_dim=NUM_LABELS, dw_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_fn


def train(model, train_dataset, node_embeddings, optimizer, loss_fn, epochs=EPOCHS):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        # DynamicGraphTemporalSignal is iterable over time snapshots
        # It returns (x, edge_index, edge_weight, y) for each t
        for time, snapshot in enumerate(train_dataset):
            deep_walk_emb = node_embeddings[time] if EMBEDDING else None
            optimizer.zero_grad()

            output = model(snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, deep_walk_emb)

            loss = loss_fn(output, snapshot.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / train_dataset.snapshot_count
        losses.append(avg_loss)
        # print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
    return losses

def evaluate(model, test_dataset, node_embeddings, loss_fn, threshold=THRESHOLD):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for time, snapshot in enumerate(test_dataset):
            deep_walk_emb = node_embeddings[time] if EMBEDDING else None
            y_hat = model(snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, deep_walk_emb)
            loss = loss_fn(y_hat, snapshot.y)
            total_loss += loss.item()

            # Apply sigmoid and threshold to get predictions
            probs = torch.sigmoid(y_hat)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_targets.append(snapshot.y.cpu())

    avg_loss = total_loss / test_dataset.snapshot_count

    # Concatenate all batches and compute metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    val_f1 = f1_score(all_targets, all_preds, average='macro')  # macro means compute per label and then average

    print(f"Test BCE Loss: {avg_loss:.4f} | Validation Macro F1 Score: {val_f1:.4f}")


if __name__ == "__main__":
    config = TemporalMultiLabelGeneratorConfig(m_rel=NUM_FEATURES, m_irr=0, m_red=0, 
                                               q=NUM_LABELS, N=NUM_NODES,
                                               max_r=0.7, 
                                               min_r=((NUM_LABELS / 10) + 1) / NUM_LABELS, 
                                               mu=0, b=0.1, alpha=16, 
                                               theta=np.pi / 7,
                                               horizon=NUM_TIMESTEPS)
    
    train_dataset, test_dataset, train_embedding, test_embedding = load_data(config)
    model, optimizer, loss_fn = initialize_model()

    losses = train(model, train_dataset, train_embedding, optimizer, loss_fn)
    evaluate(model, test_dataset, test_embedding, loss_fn)

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
