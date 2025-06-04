import numpy as np
from matplotlib import pyplot as plt

from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig, TemporalMultiLabelGenerator

alphas = range(0, 16)
intra_homophily_mean = []
intra_homophily_std = []
inter_homophily = []
for alpha in alphas:
    config = TemporalMultiLabelGeneratorConfig(m_rel=10,
                                               m_irr=10,
                                               m_red=0,
                                               q=20,
                                               N=500,
                                               max_r=0.8,
                                               min_r=((20 / 10) + 1) / 20,
                                               mu=0,
                                               b=0.05,
                                               alpha=alpha,
                                               theta=np.pi / 7,
                                               horizon=15
                                               )
    tmlg = TemporalMultiLabelGenerator(config)

    temporal_hyper_spheres = tmlg.generate()
    temporal_hyper_spheres.save_to_file(f"../data/temporal_hyper_spheres_alpha_{alpha}")
    inter_homophily.append(temporal_hyper_spheres.inter_homophily())
    temporal_intra_homophily = temporal_hyper_spheres.intra_homophily()
    intra_homophily_mean.append(np.mean(temporal_intra_homophily))
    intra_homophily_std.append(np.std(temporal_intra_homophily))

plt.figure(figsize=(8, 5))
plt.plot(alphas, inter_homophily, 's--', label='Inter-Homophily', color='orange')

plt.errorbar(alphas, intra_homophily_mean, yerr=intra_homophily_std,
             fmt='o-', capsize=5, label='Intra-Homophily (mean ± std)')
plt.xlabel("Alpha")
plt.ylabel("Homophily")
plt.title("Homophily vs Alpha")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
