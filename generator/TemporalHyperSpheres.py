import pickle
from generator import HyperSpheresGraph
from generator.utils import jaccard_similarity


class TemporalHyperSpheres:
    def __init__(self, temporal_hyper_spheres: [HyperSpheresGraph]):
        self.temporal_hyper_spheres = temporal_hyper_spheres

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.temporal_hyper_spheres, f)
        print(f"TemporalHyperSpheres saved to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            temporal_hyper_spheres = pickle.load(f)
        return cls(temporal_hyper_spheres)

    def inter_homophily(self, start_time=0, end_time=None):
        if end_time is None:
            end_time = len(self.temporal_hyper_spheres)

        total_jaccard = 0
        total_edges = 0
        for graph in self.temporal_hyper_spheres[start_time:end_time]:
            
            N = graph.edge_list.shape[-1]
            for k in range(0, N, 2):
                i, j = graph.edge_list[:, k]
                total_jaccard += jaccard_similarity(graph.y_data[i], graph.y_data[j])
            total_edges += N/2
            
        if total_edges == 0:
            return 1
        return total_jaccard / total_edges

    def intra_homophily(self, start_time=0, end_time=None):
        if end_time is None:
            end_time = len(self.temporal_hyper_spheres)

        homophily = []
        for graph in self.temporal_hyper_spheres[start_time:end_time]:
            homophily.append(graph.intra_homophily())

        return homophily
