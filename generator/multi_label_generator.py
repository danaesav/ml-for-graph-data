import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl
from dataclasses import dataclass, field
from typing import Literal

from scipy.spatial import distance

from generator.HyperSpheres import HyperSpheres
from generator.HyperSpheresGraph import HyperSpheresGraph


@dataclass
class MultiLabelGeneratorConfig:
    """
    Multi-Label Dataset Generator Configuration parameters from Tomas et al. and Zhao et al.

    Attributes
    ----------
    m_rel : int
        dimension of relevant features.
    
    m_irr : int
        dimension of irrelevant features.
    
    m_red : int
        dimension of redundant features.

    m :  int
        dimension of data sample
    
    q : float
        number of classes.
    
    N : int
        number of data samples.
    
    max_r : float
        Maximum allowed radius for sub hypersphere.
    
    min_r : float
        Minimum allowed radius for sub hypersphere.
    
    mu : float
        noise probability.
    
    alpha : float
        homophily parameter.
    
    b : float
        characteristic distance.
    """

    # Tomas et al. Parameters
    m_rel: int
    m_irr: int
    m_red: int
    m: int = field(init=False)

    q: int
    N: int

    max_r: float
    min_r: float
    mu: float

    # Zhao et al. Parameters
    b: float
    alpha: float

    def __post_init__(self):
        self.m = self.m_rel + self.m_irr + self.m_red


class MultiLabelGenerator:
    """
    Implements the synthetic multi-label dataset generator using the *HyperSpheres* strategy from `Tomas et al.`_
    

    .. _Tomas et al.: https://arxiv.org/pdf/2304.10398
    """

    def __init__(self, config):

        self.set_config(config)

    @staticmethod
    def parameter_feasibility_check(config):
        assert config.m_rel > 0
        assert config.m_irr >= 0
        assert config.m_red >= 0
        assert config.q > 0
        assert config.N > 0
        assert config.max_r > 0
        assert config.min_r > 0
        assert config.mu >= 0
        assert config.m_red <= config.m_rel
        assert config.min_r < config.max_r and config.max_r <= 0.8
        # assert config.min_r <= np.floor_divide(config.q/10 + 1 , config.q), f"{config.min_r} <= {np.floor_divide(config.q/10 + 1 , config.q)}"
        assert config.min_r <= np.divide(config.q / 10 + 1, config.q)
        assert config.b > 0
        assert config.alpha >= 0

    def set_config(self, config):
        self.parameter_feasibility_check(config)

        # Tomas et al. Parameters
        self.m_rel = config.m_rel
        self.m_irr = config.m_irr
        self.m_red = config.m_red
        self.m = config.m

        self.q = config.q
        self.N = config.N

        self.max_r = config.max_r
        self.min_r = config.min_r
        self.mu = config.mu

        # Zhao et al. Parameters
        self.b = config.b
        self.alpha = config.alpha

    def generate_hyper_spheres(self, sphere_sampling:Literal['polar', 'cartesian']='polar', data_sampling:Literal['polar', 'cartesian', 'global']='global'):

        # spheres represented as tuple(center, radius)

        # 1. generate hypersphere HS
        sphere_HS = (np.zeros(self.m_rel), 1)

        # 2. generate small hyperspheres hs
        spheres_hs = []

        for i in range(self.q):
            ci = np.zeros(self.m_rel)
            ri = np.random.uniform(self.min_r, self.max_r)

            if sphere_sampling == 'cartesian':
                max_c = 1 - ri
                min_c = -max_c

                for j in range(self.m_rel):
                    cij = np.random.uniform(min_c, max_c)

                    ci[j] = cij
                    max_c = np.sqrt(np.square(1 - ri) - np.square(ci).sum())
                    min_c = -max_c

            else:
                # sample uniformally with polar coordinates 
                # sample (l, theta_1, ... theta_{m-1})
                ai = np.random.uniform(0, np.pi, size=self.m_rel-2) # theta_(1, ..., m-2) coord shape(m - 2)
                ai = np.append(ai, np.random.uniform(0, np.pi*2)) # theta_(m-1) coord  shape(m-1)
                li = np.random.uniform(0, 1-ri) # length coord

                for j in range(self.m_rel-1):
                    ci[j] = li * np.prod(np.sin(ai[:j])) * np.cos(ai[j])

                ci[self.m_rel-1] = li * np.prod(np.sin(ai))

            spheres_hs.append((ci, ri))

        # 3. Generate N points
        # generate prob. distribution for each sphere
        ris = np.array([hs[1] for hs in spheres_hs])
        f = self.N / np.sum(ris)
        nis = np.round(ris * f)
        prob = nis / self.N
        prob /= np.sum(prob)

        x_rel = np.zeros((self.N, self.m_rel))
        for i in range(self.N):

            # select sphere hs to put point
            k = np.random.choice(range(self.q), p=prob)

            xk = np.zeros(self.m_rel)

            cij, ri = spheres_hs[k]

            if data_sampling == 'cartesian':

                max_x = cij + ri
                min_x = cij - ri

                for j in range(self.m_rel):
                    xkj = np.random.uniform(min_x, max_x)

                    xk[j] = xkj[j]
                    x_range = np.sqrt(np.square(ri) - np.square(xk[:(j + 1)] - cij[:(j + 1)]).sum())
                    max_x = cij + x_range
                    min_x = cij - x_range

            elif data_sampling == 'polar':
                # TODO: sample uniformally with polar coordinates 

                # sample (l, theta_1, ... theta_{m-1})
                ai = np.random.uniform(0, np.pi, size=self.m_rel-2) # theta_(1, ..., m-2) coord shape(m - 2)
                ai = np.append(ai, np.random.uniform(0, np.pi*2)) # theta_(m-1) coord  shape(m-1)
                li = np.random.uniform(0, ri) # length coord

                for j in range(self.m_rel-1):
                    xk[j] = li * np.prod(np.sin(ai[:j])) * np.cos(ai[j])

                xk[self.m_rel-1] = li * np.prod(np.sin(ai))

                xk += cij

            else:
                # global sampling
                # sample (l, theta_1, ... theta_{m-1})
                ai = np.random.uniform(0, np.pi, size=self.m_rel-2) # theta_(1, ..., m-2) coord shape(m - 2)
                ai = np.append(ai, np.random.uniform(0, np.pi*2)) # theta_(m-1) coord  shape(m-1)
                li = np.random.uniform(0, 1) # length coord

                for j in range(self.m_rel-1):
                    xk[j] = li * np.prod(np.sin(ai[:j])) * np.cos(ai[j])

                xk[self.m_rel-1] = li * np.prod(np.sin(ai))

            x_rel[i] = xk

        # 4. Expand to M features
        x_data = self._extend_feature(x_rel)

        hyper_spheres = HyperSpheres(x_data, spheres_hs, sphere_HS)

        return hyper_spheres

    def form_graph(self, hyper_spheres: HyperSpheres):
        # 5. Generate multi-labels (one-hot-encoded)
        # 6. noisy label flips
        y_data, y_data_noised = self._generate_labels(hyper_spheres.x_data, hyper_spheres.spheres_hs)

        # 7. compute edges
        adj_mat, edge_list = self._compute_edges(y_data)

        return HyperSpheresGraph(hyper_spheres, y_data, y_data_noised, adj_mat, edge_list)

    def _extend_feature(self, x_rel):
        # sample m_irr features
        x_irr = np.random.uniform(-1, 1, size=(self.N, self.m_irr))

        # sample m_red features
        col_idx = np.random.randint(self.m_rel, size=(self.N, self.m_red))
        row_idx = np.arange(self.N)[:, np.newaxis]
        x_red = x_rel[row_idx, col_idx]

        x_data = np.concatenate([x_rel, x_irr, x_red], axis=1)

        assert x_data.shape == (self.N, self.m)

        return x_data

    def _generate_labels(self, x_data, spheres_hs):

        # 5. Generate multi-labels (one-hot-encoded)
        y_data = np.zeros((self.N, self.q), dtype=bool)

        for i in range(self.N):
            # xi = x_rel[i]
            xi = x_data[i, :self.m_rel]

            for j in range(self.q):
                cj, rj = spheres_hs[j]

                if np.sqrt(np.square(xi - cj).sum()) <= rj:
                    y_data[i, j] = True

        # 6. noisy label flips
        flip_mask = np.random.rand(self.N, self.q) < self.mu

        # Flip using XOR (logical exclusive or)
        y_data_noised = np.logical_xor(y_data, flip_mask)

        return y_data, y_data_noised

    def _compute_edges(self, y_data):

        adj_mat = np.zeros((self.N, self.N), dtype=bool)
        edge_list = []

        for i in range(self.N):
            for j in range(i):

                pij = self.edge_prob(y_data[i], y_data[j])

                prob = random.uniform(0, 1)
                if prob <= pij:
                    adj_mat[i, j] = True
                    adj_mat[j, i] = True
                    edge_list.append((i, j))

        return adj_mat, edge_list

    def edge_prob(self, yi, yj):
        """
        implements equation (2) from `Zhao et al.`_

        .. _Zhao et al.: https://arxiv.org/pdf/2304.10398

        """
        dist = distance.hamming(yi, yj)
        p = 1 / (2 * (1 + pow(pow(self.b, -1) * dist, self.alpha)))
        # p = 1 / (10 * (1 + pow(pow(self.b, -1) * dist, self.alpha)))

        return p

    def visualize(self, hyper_spheres_graph: HyperSpheresGraph, ax=None, legend=True):
        """
        x_data : array(N, M)
        """

        assert self.m_rel == 2  # only plot if 2d

        if ax is None:
            fig, ax = plt.subplots()
            plot = True
        else:
            plot = False

        ax.scatter(hyper_spheres_graph.hyper_spheres.x_data[:, 0], hyper_spheres_graph.hyper_spheres.x_data[:, 1],
                   c='black', s=10, label='Data Points',
                   zorder=3)

        # plot edges
        for e1, e2 in hyper_spheres_graph.edge_list:
            x_values = [hyper_spheres_graph.hyper_spheres.x_data[e1, 0],
                        hyper_spheres_graph.hyper_spheres.x_data[e2, 0]]
            y_values = [hyper_spheres_graph.hyper_spheres.x_data[e1, 1],
                        hyper_spheres_graph.hyper_spheres.x_data[e2, 1]]
            ax.plot(x_values, y_values, 'k-', linewidth=1)

        cmap = mpl.colormaps['Paired']  # or 'tab20', 'Set1', etc.

        circle_HS = Circle(tuple(hyper_spheres_graph.hyper_spheres.sphere_HS[0]),
                           hyper_spheres_graph.hyper_spheres.sphere_HS[1], color='gray', fill=False,
                           linewidth=3)
        ax.add_patch(circle_HS)

        for i, ((x, y), r) in enumerate(hyper_spheres_graph.hyper_spheres.spheres_hs):
            color = cmap(i % cmap.N)  # loop through colormap if more than N
            circle = Circle((x, y), r, color=color, fill=False, linewidth=2, label=f'Circle {i + 1}')
            ax.add_patch(circle)

        ax.set_aspect('equal', 'box')

        if legend:
            ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

        if plot:
            plt.show()


if __name__ == "__main__":
    config = MultiLabelGeneratorConfig(m_rel=2,
                                       m_irr=0,
                                       m_red=0,
                                       q=5,
                                       N=15,
                                       max_r=0.8,
                                       min_r=0.3,
                                       mu=0,
                                       b=0.1,
                                       alpha=16)

    mlg = MultiLabelGenerator(config)

    hyper_spheres = mlg.generate_hyper_spheres()
    hyper_spheres_graph = mlg.form_graph(hyper_spheres)
    mlg.visualize(hyper_spheres_graph)
