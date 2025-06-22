import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from generator.HyperSpheres import HyperSpheres
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig, TemporalMultiLabelGenerator

REPEATS = 3
NUM_NODES = 500
HORIZON = 15
NUM_LABELS = 20
NUM_FEATURES = 10
alphas = range(0, 11)


SAMPLING = [['cartesian', 'cartesian'],
            ['cartesian', 'polar'],
            ['cartesian', 'global'],
            ['polar', 'cartesian'],
            ['polar', 'polar'],
            ['polar', 'global']]

# 'cartesian', 'global'


intra_homophily_mean = [[[] for r in range(REPEATS)] for l in range(len(SAMPLING))]
intra_homophily_std = [[[] for r in range(REPEATS)] for l in range(len(SAMPLING))]
inter_homophily = [[[] for r in range(REPEATS)] for l in range(len(SAMPLING))]
intra_homophily = [[[[] for i in range(HORIZON + 1)] for r in range(REPEATS)] for l in range(len(SAMPLING))]

inter_homophily_mean = [[] for l in range(len(SAMPLING))]
inter_homophily_std = [[] for l in range(len(SAMPLING))]

intra_homophily_std_mean = [[] for l in range(len(SAMPLING))]
intra_homophily_std_std = [[] for l in range(len(SAMPLING))]

label = []

for k, setting in enumerate(SAMPLING):
    sphere_sampling = setting[0]
    data_sampling = setting[1]

    label.append(f'sphere:{sphere_sampling}, data:{data_sampling}')

    for r in tqdm(range(REPEATS)):

        tmgc_config = TemporalMultiLabelGeneratorConfig(m_rel=NUM_FEATURES,
                                                        m_irr=0,
                                                        m_red=0,
                                                        q=NUM_LABELS,
                                                        N=NUM_NODES,
                                                        max_r=0.7,
                                                        min_r=((NUM_LABELS / 10) + 1) / NUM_LABELS,
                                                        mu=0,
                                                        b=0.05,
                                                        alpha=alphas[0],
                                                        theta=np.pi / (HORIZON//2),
                                                        horizon=HORIZON,
                                                        sphere_sampling=sphere_sampling,
                                                        data_sampling=data_sampling,
                                                        rotation_reference='data'
                                                        )
        tmlg = TemporalMultiLabelGenerator(tmgc_config)



        hyper_spheres_base = tmlg.generate_hyper_spheres()

        for alpha in alphas:
            tmlg.alpha = alpha

            temporal_hyper_spheres = tmlg.generate(hyper_spheres_base)
            inter_homophily[k][r].append(temporal_hyper_spheres.inter_homophily())
            temporal_intra_homophily = temporal_hyper_spheres.intra_homophily()
            intra_homophily_mean[k][r].append(np.mean(temporal_intra_homophily))
            intra_homophily_std[k][r].append(np.std(temporal_intra_homophily))
            for h in range(HORIZON + 1):
                intra_homophily[k][r][h].append(temporal_intra_homophily[h])



    inter_homophily_mean[k] = np.mean(inter_homophily[k], axis=0)
    inter_homophily_std[k] = np.std(inter_homophily[k], axis=0)

    intra_homophily_std_mean[k] = np.mean(intra_homophily_std[k], axis=0)
    intra_homophily_std_std[k] = np.std(intra_homophily_std[k], axis=0)

colors = plt.cm.viridis(np.linspace(0, 1, HORIZON + 1))
plt.figure(figsize=(8, 5))
for k in range(len(label)):
    plt.errorbar(alphas, inter_homophily_mean[k], yerr=inter_homophily_std[k],
                fmt='o-', capsize=5, label=label[k])
    
plt.xlabel("Alpha")
plt.ylim(0, 1)
plt.ylabel("Inter Homophily (mean ± std)")
plt.title(f"{NUM_LABELS} labels, {NUM_FEATURES} features")
plt.grid(True)
plt.legend(title='Sampling Techniques')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Sampling Techniques')
plt.tight_layout()
plt.show(block=False)


plt.figure(figsize=(8, 5))
for k in range(len(label)):
    plt.errorbar(alphas, intra_homophily_std_mean[k], yerr=intra_homophily_std_std[k],
                fmt='o-', capsize=5, label=label[k])
    
plt.xlabel("Alpha")
# plt.ylim(0, 1)
plt.ylabel("Intra Homophily Std (mean ± std)")
plt.title(f"{NUM_LABELS} labels, {NUM_FEATURES} features")
plt.grid(True)
plt.legend(title='Sampling Techniques')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Sampling Techniques')
plt.tight_layout()
plt.show()
