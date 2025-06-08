from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal

from generator.HyperSpheres import HyperSpheres
from generator.HyperSpheresGraph import HyperSpheresGraph
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
    rotation_reference:Literal['data', 'sphere']


class TemporalMultiLabelGenerator(MultiLabelGenerator):

    def __init__(self, config):
        super().__init__(config)

    def set_config(self, config):
        super().set_config(config)

        self.theta = config.theta
        self.horizon = config.horizon
        self.rotation_reference:Literal['data', 'sphere'] = config.rotation_reference

    @staticmethod
    def parameter_feasibility_check(config):

        MultiLabelGenerator.parameter_feasibility_check(config)

    def generate(self, hyper_spheres: HyperSpheres = None):
        # generate time zero data
        if hyper_spheres is None:
            hyper_spheres = super().generate_hyper_spheres()

        base_graph = super().form_graph(hyper_spheres)

        temporal_hyper_spheres = [base_graph]

        if self.horizon == 0:
            return TemporalHyperSpheres(temporal_hyper_spheres)

        # generate temporal changes

        # compute rotational matrix
        x_rel = base_graph.hyper_spheres.x_data[:, :self.m_rel]  # (N, M')

        # use hypersphere centers for rotation matrix instead
        if self.rotation_reference == 'sphere':
            X_center = np.array([ci for (ci, ri) in base_graph.hyper_spheres.spheres_hs]) #(q, 2)
            R = self.rotational_matrix(X_center)

        else:
            R = self.rotational_matrix(x_rel)

        for t in range(self.horizon):
            # 1. rotate data
            # x_rel = self.translation(x_rel)   # translation instead of rotation
            x_rel = x_rel @ R  # x_rel to be used in next time step
            xt = self._extend_feature(x_rel)
            hyper_spheres_t = HyperSpheres(xt, base_graph.hyper_spheres.spheres_hs, base_graph.hyper_spheres.sphere_HS)

            hyper_spheres_graph_t = super().form_graph(hyper_spheres_t)

            temporal_hyper_spheres.append(
                hyper_spheres_graph_t)

        return TemporalHyperSpheres(temporal_hyper_spheres)

    def translation(self, x_rel):
        random_noise = np.random.uniform(-0.005, 0.005, size=x_rel.shape)
        x_t = x_rel  + random_noise
        return x_t

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
                                               horizon=15,
                                               sphere_sampling='polar',
                                               data_sampling='global',
                                               rotation_reference='data'
                                               )

    tmlg = TemporalMultiLabelGenerator(config)

    temporal_hyper_spheres = tmlg.generate()
    tmlg.visualize(temporal_hyper_spheres)
