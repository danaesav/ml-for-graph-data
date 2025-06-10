import torch
import torch.nn as nn
from torch import optim
from torch_geometric_temporal import temporal_signal_split
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from dataset_loader import DatasetLoader
from models.MultiLabelEvolveGCN import MultiLabelEvolveGCN
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig


NUM_NODES = 500  # Must match N in generator config
NUM_REL_FEATURES = 10
NUM_IRR_FEATURES = 10
NUM_RED_FEATURES = 0
NUM_LABELS = 20  # q = number of hyperspheres
NUM_TIMESTEPS = 15  # horizon
EPOCHS = 200
LR = 1e-2
THRESHOLD = 0.5  # for classification


def load_data(config, train_ratio=0.6):
    dataset, embeddings = DatasetLoader(config).get_dataset()
    train, val_test = temporal_signal_split(dataset, train_ratio=train_ratio)
    val, test = temporal_signal_split(val_test, train_ratio=(1-train_ratio)/2.0)
    return train, val, test


def initialize_model():
    model = MultiLabelEvolveGCN(
        num_nodes=NUM_NODES,
        node_features=NUM_REL_FEATURES + NUM_IRR_FEATURES + NUM_RED_FEATURES,
        num_labels=NUM_LABELS
    )
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    return model, optimizer, loss_fn

def _eval_rocauc(y_true, y_pred):
    rocauc_list = []

    for i in range(y_true.shape[1]):

        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

def train(model, train_dataset, optimizer, loss_fn, epochs=EPOCHS):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for snapshot in train_dataset:
            optimizer.zero_grad()
            y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            loss = loss_fn(y_hat, snapshot.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / train_dataset.snapshot_count
        losses.append(avg_loss)

    return losses

def evaluate(model, test_dataset, loss_fn, threshold=THRESHOLD):
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
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_targets.append(snapshot.y.cpu())

    avg_loss = total_loss / test_dataset.snapshot_count

    # Concatenate all batches and compute metrics
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    val_f1_macro = f1_score(all_targets, all_preds, average='macro')  # macro means compute per label and then average
    val_f1_micro = f1_score(all_targets, all_preds, average='micro')
    val_ap_macro = average_precision_score(y_true = all_targets, y_score = all_preds, average='macro')
    val_roc_auc_score = _eval_rocauc(all_targets, all_preds)

    print(f"Test BCE Loss: {avg_loss:.4f}\n"
          f"Validation Macro F1 Score: {val_f1_macro:.4f}\n"
          f"Validation Micro F1 Score: {val_f1_micro:.4f}\n"
          f"Validation Macro Average Precision Score: {val_ap_macro:.4f}\n"
          f"Validation AUC-ROC Score: {val_roc_auc_score:.4f}\n")


if __name__ == "__main__":
    config = TemporalMultiLabelGeneratorConfig(m_rel=NUM_REL_FEATURES,
                                               m_irr=NUM_IRR_FEATURES,
                                               m_red=NUM_RED_FEATURES,
                                               q=NUM_LABELS, 
                                               N=NUM_NODES,
                                               max_r=0.7, 
                                               min_r=0.1, 
                                               mu=0, 
                                               b=0.12   ,
                                               alpha=8.8,
                                               theta=np.pi / 7,
                                               horizon=NUM_TIMESTEPS,
                                               sphere_sampling='polar',
                                               data_sampling='global',
                                               rotation_reference='data',
                                               )
    train_dataset, val_dataset, test_dataset = load_data(config)
    model, optimizer, loss_fn = initialize_model()

    losses = train(model, train_dataset, optimizer, loss_fn)
    evaluate(model, val_dataset, loss_fn)


    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
