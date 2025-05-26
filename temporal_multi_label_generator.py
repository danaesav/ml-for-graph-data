import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib as mpl
from dataclasses import dataclass, field
from multi_label_generator import MultiLabelGenerator, MultiLabelGeneratorConfig

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


    def generate(self):
        # generate time zero data
        x_data, y_data, y_data_noised, spheres_hs, sphere_HS, adj_mat, edge_list = super().generate()

        # temporal part
        tx_data = np.zeros((self.horizon+1, *x_data.shape))
        ty_data = np.zeros((self.horizon+1, *y_data.shape))
        ty_data_noised = np.zeros((self.horizon+1, *y_data_noised.shape))
        t_adj_mat = np.zeros((self.horizon+1, *adj_mat.shape))
        t_edge_list = []

        #init time 0 data
        tx_data[0] = x_data
        ty_data[0] = y_data
        ty_data_noised[0] = y_data_noised
        t_adj_mat[0] = adj_mat
        t_edge_list.append(edge_list)

        # generate temporal changes

        # compute rotational matrix
        x_rel = x_data[:, :self.m_rel] #(N, M')
        R = self.rotational_matrix(x_rel)

        for t in range(1, self.horizon+1):
            # 1. rotate data
            x_rel = x_rel @ R # x_rel to be used in next time step
            xt = self._extend_feature(x_rel)

            tx_data[t] = xt
            
            # 2. regenerate labels
            yt, yt_noised = self._generate_labels(xt, spheres_hs)

            ty_data[t] = yt
            ty_data_noised[t] = yt_noised

            # 3. recompute edges
            adj_mat_t, edge_list_t = self._compute_edges(yt)

            t_adj_mat[t] = adj_mat_t
            t_edge_list.append(edge_list_t)
            

        return tx_data, ty_data, ty_data_noised, spheres_hs, sphere_HS, t_adj_mat, t_edge_list

    def rotational_matrix(self, X):
        """
        X : array(N, M)
        """
        # 1. retrieve 2D orthonormal basis for rotational plane via PCA -> use two most informative dimensions
        X_c = X - np.mean(X, axis=0) #(N, M')
        X_cov = np.cov(X_c, rowvar=False) # (M', M')
        eigenvalues, eigenvectors = np.linalg.eigh(X_cov)
        #sort
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # orthonormal basis due to eigh() with symmetric cov matrix
        U = eigenvectors[:, :2] #(M', 2) 

        # 2. compute rotational matrix for relvant features
        # R_m' = U R_2 U' + (I - U U.')
        R_2 = np.array([[np.cos(self.theta), -np.sin(self.theta)], 
                        [np.sin(self.theta),  np.cos(self.theta)]])
        R_m = U @ R_2 @ U.T + (np.eye(self.m_rel) - U @ U.T)

        return R_m


    def visualize(self, sphere_HS, spheres_hs, tx_data, t_edge_list, ax=None):

        rows = int(np.floor(np.sqrt(self.horizon+1)))
        cols = int(np.ceil((self.horizon+1)/float(rows)))

        fig, axs = plt.subplots(rows, cols)
        fig.tight_layout()

        for i, axi in enumerate(axs.flat):

            super().visualize(sphere_HS, spheres_hs, tx_data[i], t_edge_list[i], axi, False)
            axi.set_title(f'time={i}')

            if i == self.horizon:
                axi.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                break

        plt.show()



if __name__ == "__main__":

    config = TemporalMultiLabelGeneratorConfig(m_rel=2, 
                                               m_irr=0, 
                                               m_red=0 ,
                                               q=5, 
                                               N=15, 
                                               max_r=0.7, 
                                               min_r=0.1, 
                                               mu=0, 
                                               b=0.1, 
                                               alpha=16,
                                               theta=np.pi/7,
                                               horizon=15
                                               )

    tmlg = TemporalMultiLabelGenerator(config)

    tx_data, ty_data, ty_data_noised, spheres_hs, sphere_HS, t_adj_mat, t_edge_list  = tmlg.generate()

    tmlg.visualize(sphere_HS, spheres_hs, tx_data, t_edge_list)