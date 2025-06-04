import pickle

from generator.utils import jaccard_similarity


class HyperSpheres:
    def __init__(self, x_data, y_data, y_data_noised, spheres_hs, sphere_HS, adj_mat, edge_list):
        self.x_data = x_data
        self.y_data = y_data.astype(float)
        self.y_data_noised = y_data_noised.astype(float)
        self.spheres_hs = spheres_hs
        self.sphere_HS = sphere_HS
        self.adj_mat = adj_mat
        self.edge_list = edge_list

    def save_to_file(self, filename):
        data = {
            'x_data': self.x_data,
            'y_data': self.y_data,
            'y_data_noised': self.y_data_noised,
            'spheres_hs': self.spheres_hs,
            'sphere_HS': self.sphere_HS,
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
        for i, j in self.edge_list:
            total += jaccard_similarity(self.y_data[i], self.y_data[j])
        return total / len(self.edge_list)
