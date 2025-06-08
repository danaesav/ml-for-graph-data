import os

import numpy as np
from matplotlib import pyplot as plt

from generator.HyperSpheres import HyperSpheres
from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig, TemporalMultiLabelGenerator

HORIZON = 3
NUM_LABELS = 20
NUM_FEATURES = 10
alphas = range(0, 11)
filename = f"./data/base_hyper_spheres_labels{NUM_LABELS}_features{NUM_FEATURES}"

intra_homophily_mean = []
intra_homophily_std = []
inter_homophily = []
intra_homophily = []
for h in range(HORIZON + 1):
    intra_homophily.append([])

tmgc_config = TemporalMultiLabelGeneratorConfig(m_rel=NUM_FEATURES,
                                                m_irr=0,
                                                m_red=0,
                                                q=NUM_LABELS,
                                                N=500,
                                                max_r=0.8,
                                                min_r=((NUM_LABELS / 10) + 1) / NUM_LABELS,
                                                mu=0,
                                                b=0.05,
                                                alpha=alphas[0],
                                                theta=np.pi / 7,
                                                horizon=HORIZON,
                                                sphere_sampling='polar',
                                                data_sampling='global',
                                                rotation_reference='data'
                                                )
tmlg = TemporalMultiLabelGenerator(tmgc_config)
if os.path.exists(filename):
    hyper_spheres_base = HyperSpheres.load_from_file(filename)
    print("Loaded existing file.")
else:
    hyper_spheres_base = tmlg.generate_hyper_spheres()
    hyper_spheres_base.save_to_file(filename)
    print("File not found. Generated and saved new data.")

for alpha in alphas:
    tmlg.alpha = alpha

    temporal_hyper_spheres = tmlg.generate(hyper_spheres_base)
    # temporal_hyper_spheres.save_to_file(f"../data/temporal_hyper_spheres_alpha_{alpha}")
    inter_homophily.append(temporal_hyper_spheres.inter_homophily())
    temporal_intra_homophily = temporal_hyper_spheres.intra_homophily()
    intra_homophily_mean.append(np.mean(temporal_intra_homophily))
    intra_homophily_std.append(np.std(temporal_intra_homophily))
    for h in range(HORIZON + 1):
        intra_homophily[h].append(temporal_intra_homophily[h])

colors = plt.cm.viridis(np.linspace(0, 1, HORIZON + 1))
plt.figure(figsize=(8, 5))
for i in range(HORIZON + 1):
    plt.plot(alphas, intra_homophily[i], 's--', label=f'Intra-Homophily time {i}', color=colors[i])

plt.plot(alphas, inter_homophily, 's--', label='Inter-Homophily', color='red')
# plt.errorbar(alphas, intra_homophily_mean, yerr=intra_homophily_std,
#              fmt='o-', capsize=5, label='Intra-Homophily (mean ± std)')
plt.xlabel("Alpha")
plt.ylim(0, 1)
plt.ylabel("Homophily")
plt.title(f"Rotation {NUM_LABELS} labels, {NUM_FEATURES} features")
# plt.title(f"Translation {NUM_LABELS} labels, {NUM_FEATURES} features")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
