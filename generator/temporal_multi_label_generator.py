import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from generator.HyperSpheres import HyperSpheres
from generator.TemporalHyperSpheres import TemporalHyperSpheres
from generator.multi_label_generator import MultiLabelGenerator, MultiLabelGeneratorConfig


@dataclass
class TemporalMultiLabelGeneratorConfig(MultiLabelGeneratorConfig):
    """
    Extended MultiLabelGeneratorConfig class with temporal support.

    Inherits from `MultiLabelGeneratorConfig`.

    Additional Attributes
    ---------------------
    theta : float
        temporal parameter.

    horizon : int
        time horizon
    """

    theta: float
    horizon: int


class TemporalMultiLabelGenerator(MultiLabelGenerator):

    def __init__(self, config):
        super().__init__(config)

    def set_config(self, config):
        super().set_config(config)

        self.theta = config.theta
        self.horizon = config.horizon

    @staticmethod
    def parameter_feasibility_check(config):

        MultiLabelGenerator.parameter_feasibility_check(config)

        assert config.horizon > 0

    def generate(self, base_graph: HyperSpheres = None):
        # generate time zero data
        if base_graph is None:
            base_graph = super().generate()

        temporal_hyper_spheres = [base_graph]

        # generate temporal changes

        # compute rotational matrix
        x_rel = base_graph.x_data[:, :self.m_rel]  # (N, M')
        R = self.rotational_matrix(x_rel)

        for t in range(1, self.horizon + 1):
            # 1. rotate data
            x_rel = x_rel @ R  # x_rel to be used in next time step
            xt = self._extend_feature(x_rel)

            # 2. regenerate labels
            yt, yt_noised = self._generate_labels(xt, base_graph.spheres_hs)

            # 3. recompute edges
            adj_mat_t, edge_list_t = self._compute_edges(yt)

            temporal_hyper_spheres.append(
                HyperSpheres(xt, yt, yt_noised, base_graph.spheres_hs, base_graph.sphere_HS, adj_mat_t, edge_list_t))

        return TemporalHyperSpheres(temporal_hyper_spheres)

    def rotational_matrix(self, X):
        """
        X : array(N, M)
        """
        # 1. retrieve 2D orthonormal basis for rotational plane via PCA -> use two most informative dimensions
        X_c = X - np.mean(X, axis=0)  # (N, M')
        X_cov = np.cov(X_c, rowvar=False)  # (M', M')
        eigenvalues, eigenvectors = np.linalg.eigh(X_cov)
        # sort
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # orthonormal basis due to eigh() with symmetric cov matrix
        U = eigenvectors[:, :2]  # (M', 2)

        # 2. compute rotational matrix for relvant features
        # R_m' = U R_2 U' + (I - U U.')
        R_2 = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                        [np.sin(self.theta), np.cos(self.theta)]])
        R_m = U @ R_2 @ U.T + (np.eye(self.m_rel) - U @ U.T)

        return R_m

    def visualize(self, ths: TemporalHyperSpheres, ax=None):

        rows = int(np.floor(np.sqrt(self.horizon + 1)))
        cols = int(np.ceil((self.horizon + 1) / float(rows)))

        fig, axs = plt.subplots(rows, cols)
        fig.tight_layout()

        for i, axi in enumerate(axs.flat):

            super().visualize(ths.temporal_hyper_spheres[i], axi, False)
            axi.set_title(f'time={i}')

            if i == self.horizon:
                axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                break

        plt.show()


if __name__ == "__main__":
    config = TemporalMultiLabelGeneratorConfig(m_rel=2,
                                               m_irr=0,
                                               m_red=0,
                                               q=5,
                                               N=15,
                                               max_r=0.7,
                                               min_r=0.1,
                                               mu=0,
                                               b=0.1,
                                               alpha=16,
                                               theta=np.pi / 7,
                                               horizon=15
                                               )

    tmlg = TemporalMultiLabelGenerator(config)

    temporal_hyper_spheres = tmlg.generate()
    tmlg.visualize(temporal_hyper_spheres)
