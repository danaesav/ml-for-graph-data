import numpy as np
import torch as th
import os
from datetime import datetime
from typing import Literal
from torch_geometric_temporal import DynamicGraphTemporalSignal
from generator.temporal_multi_label_generator import TemporalMultiLabelGenerator, TemporalMultiLabelGeneratorConfig
from models.NodeEmbedding import NodeEmbedding
from generator.HyperSpheres import HyperSpheres

class DatasetLoader(object):
    def __init__(self, config: TemporalMultiLabelGeneratorConfig, embedding_dim, embedding_method:Literal['Node2Vec', 'DeepWalk', 'Node2Vec Recurrent']='Node2Vec', filename=None, load=None):


        self.generator = TemporalMultiLabelGenerator(config)

        if load and os.path.exists(load):
            
            self.base_hypersphere = HyperSpheres.load_from_file(load)
            print("Loaded existing base file.")

        else:
            self.base_hypersphere = self.generator.generate_hyper_spheres()

            if filename:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.base_hypersphere.save_to_file(filename+'_'+timestamp)
                print("Generated and saved new base file.")


        self.lags = config.horizon+1
        self.embedder = NodeEmbedding(embedding_dim=embedding_dim)
        self.embedding_method = embedding_method

    
    # def generate_node_embeddings(self, edge_index):
    #     #Generate node embeddings for each snapshot

    #     # temporal_node_embeddings = th.zeros((self.lags, )) #(time, nodes, embedding_dim)
    #     embedding_t = self.embedder.get_embedding(edge_index, 'Node2Vec')

    #     return embedding_t




    def get_dataset(self) -> DynamicGraphTemporalSignal:
        # Generate temporal data
        ths = self.generator.generate(self.base_hypersphere)

        # if filename:
        #     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #     ths.save_to_file(filename+'_'+timestamp)

        inter_homophily = ths.inter_homophily()
        intra_homophily = ths.intra_homophily()

        # Convert edges + data into DynamicGraphTemporalSignal format
        edge_indices = []
        edge_weights = []
        features = []
        targets = []
        embeddings = []

        for t in range(len(ths.temporal_hyper_spheres)):
            # --- Edges ---
            edge_index_t = ths.temporal_hyper_spheres[t].edge_list # shape [2, num_edges]
            # print(edge_index_t.shape)
            # edge_index_t = np.array(edges_t).T  
            edge_indices.append(edge_index_t)
            adj_mat_t = ths.temporal_hyper_spheres[t].adj_mat

            # --- Edge weights (optional, here we just use 1.0) ---
            # edge_weights.append(np.zeros(edge_index_t.shape[1]))
            edge_weights.append(np.zeros(edge_index_t.size//2))

            # --- Node features and labels ---
            features.append(ths.temporal_hyper_spheres[t].hyper_spheres.x_data)
            targets.append(ths.temporal_hyper_spheres[t].y_data)

            # --- Node embedding ---
            embedding_t = self.embedder.get_embedding(edge_index_t, adj_mat_t, self.embedding_method)
            # print(edge_index_t.shape, adj_mat_t.shape ,embedding_t.shape)
            embeddings.append(embedding_t)

        # Create the PyTorch Geometric Temporal dataset
        dataset = DynamicGraphTemporalSignal(
            edge_indices=edge_indices,
            edge_weights=edge_weights,
            features=features,
            targets=targets
        )

        return dataset, embeddings, inter_homophily, intra_homophily
