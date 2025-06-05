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
        ths = self.generator.generate()

        # Convert edges + data into DynamicGraphTemporalSignal format
        edge_indices = []
        edge_weights = []
        features = []
        targets = []

        for t in range(len(ths.temporal_hyper_spheres)):
            # --- Edges ---
            edges_t = ths.temporal_hyper_spheres[t].edge_list
            edge_index_t = np.array(edges_t).T  # shape [2, num_edges]
            edge_indices.append(edge_index_t)

            # --- Edge weights (optional, here we just use 1.0) ---
            edge_weights.append(np.zeros(edge_index_t.shape[1]))

            # --- Node features and labels ---
            features.append(ths.temporal_hyper_spheres[t].hyper_spheres.x_data)
            targets.append(ths.temporal_hyper_spheres[t].y_data)

        # Create the PyTorch Geometric Temporal dataset
        dataset = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets
        )
        return dataset
