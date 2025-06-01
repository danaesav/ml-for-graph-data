import numpy as np
from torch_geometric_temporal import DynamicGraphTemporalSignal

from generator.temporal_multi_label_generator import TemporalMultiLabelGenerator, TemporalMultiLabelGeneratorConfig


class DatasetLoader(object):
    def __init__(self, config: TemporalMultiLabelGeneratorConfig):
        self.generator = TemporalMultiLabelGenerator(config)
        self.lags = config.horizon

    # TODO: def generate_deepwalk_embeddings(self):
    #     Generate deepwalk embeddings for each snapshot

    def get_dataset(self) -> DynamicGraphTemporalSignal:
        # Generate temporal data
        #   tx_data:          np.ndarray of shape (T, N, feat_dim)
        #   ty_data:          np.ndarray of shape (T, N, num_labels)
        #   t_edge_list:      list of length T; each entry is a Python list of (i,j) tuples
        tx_data, ty_data, _, _, _, _, t_edge_list = self.generator.generate()

        # Convert edges + data into DynamicGraphTemporalSignal format
        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        for t in range(len(tx_data)):
            # --- Edges ---
            edges_t = t_edge_list[t]
            edge_index_t = np.array(edges_t).T  # shape [2, num_edges]
            edge_indices.append(edge_index_t)

            # --- Edge weights (optional, here we just use 1.0) ---
            edge_weights.append(np.zeros(edge_index_t.shape[1]))

            # --- Node features and labels ---
            features.append(tx_data[t])
            targets.append(ty_data[t])

        # Create the PyTorch Geometric Temporal dataset
        dataset = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets
        )
        return dataset
