import pickle

import numpy as np

from generator import HyperSpheres
from generator.utils import jaccard_similarity
import seaborn as sns
import matplotlib.pyplot as plt


class HyperSpheresGraph:
    def __init__(self, hyperSpheres: HyperSpheres, y_data, y_data_noised, adj_mat, edge_list):
        self.hyper_spheres = hyperSpheres
        self.y_data = y_data.astype(float)
        self.y_data_noised = y_data_noised.astype(float)
        self.adj_mat = adj_mat
        self.edge_list = edge_list

    def save_to_file(self, filename):
        data = {
            'hyper_spheres': self.hyper_spheres,
            'y_data': self.y_data,
            'y_data_noised': self.y_data_noised,
            'adj_mat': self.adj_mat,
            'edge_list': self.edge_list
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(**data)

    def intra_homophily(self):
        total = 0
        if len(self.edge_list) == 0:
            return 1
        
        N = self.edge_list.shape[-1]
        for k in range(0, N, 2):
            i, j = self.edge_list[:, k]
            total += jaccard_similarity(self.y_data[i], self.y_data[j])
        return total / (N / 2)

    def plot_label_distribution(self, title=None):
        counts = self.y_data.sum(axis=-1)
        sns.violinplot(data=counts, orient='h', inner='box')
        plt.xlabel('Label count per node')

        if title:
            plt.title(title)
        plt.show()