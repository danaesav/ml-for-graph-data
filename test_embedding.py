import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from torch import optim, nn
from torch_geometric_temporal import temporal_signal_split
from tqdm import tqdm
from dataset_loader import DatasetLoader
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig
from models.TemporalMultiFix import TemporalMultiFix
from utils import metrics
import pickle

NUM_NODES = 50          # Must match N in generator config
NUM_FEATURES = 10        # m_rel = 2, total features = m_rel + m_irr + m_red = 2
NUM_LABELS = 20          # q = number of hyperspheres
NUM_TIMESTEPS = 15      # horizon
HIDDEN_DIM = 64
EPOCHS = 750
LR = 8e-3
THRESHOLD = 0.5         # for classification
EMBEDDING = True
EMBEDDING_DIM = 64
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

param = {'NUM_NODES' : 3000,  # Must match N in generator config
        'NUM_REL_FEATURES' : 10,
        'NUM_IRR_FEATURES' : 10,
        'NUM_RED_FEATURES' : 0,
        'NUM_LABELS' : 20,  # q = number of hyperspheres
        'NUM_TIMESTEPS' : 30,  # horizon
        'EPOCHS' : 500,
        'LR_MLEGCN': 2e-2,
        'LR_TMF': 8e-3,
        'THRESHOLD' : 0.5,  # for classification
        'REPEATS' : 5,
        'ALPHA' : 5,
        'EMBEDDING_DIM' : 16,
        'TRAIN_RATIO' : 0.6,
        'VALIDATION_RATIO' : 0.2,
        'TEST_RATIO' : 0.2,
        'FILENAME' : '.\\data\\base_graphs',
        'BASEFILE' : '.\\data\\base_graphs_2025-06-12_02-49-34', #None for new base graph
        'EXPERIMENT_PATH': '.\\data',
        'DATA_FILE': 'results',
        'IMAGE_FILE': 'image',
    }

edge_count = [[] for i in range(7)]
labelless_ratio = [[] for i in range(7)]


for i, alpha in tqdm(enumerate([0, 0.5, 1, 1.5, 2, 2.5, 3])):

    config = TemporalMultiLabelGeneratorConfig(m_rel=param["NUM_REL_FEATURES"],
                                                    m_irr=param["NUM_IRR_FEATURES"],
                                                    m_red=param["NUM_RED_FEATURES"],
                                                    q=param["NUM_LABELS"],
                                                    N=param["NUM_NODES"],
                                                    max_r=0.8,
                                                    min_r=((param["NUM_LABELS"] / 10) + 1) / param["NUM_LABELS"],
                                                    mu=0,
                                                    b=0.05,
                                                    alpha=alpha,
                                                    theta=np.pi / -(param["NUM_TIMESTEPS"]//-2), #ceiling division
                                                    horizon=param["NUM_TIMESTEPS"],
                                                    sphere_sampling='polar',
                                                    data_sampling='global',
                                                    rotation_reference='data',
                                                    )


    dl = DatasetLoader(config, param["EMBEDDING_DIM"], param["FILENAME"], param["BASEFILE"])


    ths = dl.generator.generate(dl.base_hypersphere)

    for t in range(len(ths.temporal_hyper_spheres)):

        edge_index_t = ths.temporal_hyper_spheres[t].edge_list # shape [2, num_edges]
        adj_mat_t = ths.temporal_hyper_spheres[t].adj_mat

        edge_count[i].append(edge_index_t.shape[-1]/2)

        iso_nodes = np.where(adj_mat_t.sum(axis=-1) == 0)[0]
        labelless_ratio[i].append(iso_nodes.size / param['NUM_NODES'])

with open('count_list', 'wb') as f:
    pickle.dump(edge_count, f)

with open('ratio_list', 'wb') as f:
    pickle.dump(labelless_ratio, f)

plt.figure(dpi=600)
for i, alpha in enumerate([0, 0.5, 1, 1.5, 2, 2.5, 3]):
    plt.plot(range(len(ths.temporal_hyper_spheres)), edge_count[i], label=f'{alpha}')

plt.xlabel('time step')
plt.ylabel('num. edges')
plt.legend(title='alpha')
plt.show(block=False)

plt.figure()

for i, alpha in enumerate([0, 0.5, 1, 1.5, 2, 2.5, 3]):
    plt.plot(range(len(ths.temporal_hyper_spheres)), labelless_ratio[i], label=f'{alpha}')

plt.xlabel('time step')
plt.ylabel('ratio of unlabeled nodes')
plt.legend(title='alpha')
plt.show()