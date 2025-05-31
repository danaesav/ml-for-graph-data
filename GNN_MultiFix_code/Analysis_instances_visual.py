# plot the majority vote in different hops away neighbors
import matplotlib.pyplot as plt
import numpy as np

# values to plot
blog_all = [[0.27758352758352756, 0.4227328537673365, 0.48182778182778185, 0.35198135198135194], # node 867
            [0.3532134532134532, 0.3033396286510858, 0.2998726582358296, 0.28671328671328666], # node 5128
            [0.3573316498316499, 0.35530985530985526, 0.2973095058095058] # node 4842
            ]
blog = [[0.32494172494172496, 0.42749388221257556, 0.4830932698948597, 0.35198135198135194],
        [0.3213869463869463, 0.31920465033658435, 0.30354785034210874, 0.28205128205128205],
        [0.38297763639868904, 0.3660971638794219, 0.2995367785367785]]


pcg_all = [[0.9935897435897437, 1.0, 0.9688131313131315, 0.9190145502645503, 0.8926406926406928, 0.9], # node 1880
           [0.8683851502033322, 0.8791292041292043, 0.8496821360457725, 0.8416240325331236, 0.8573593073593073, 0.7943722943722944], # node 963
           [0.8927817132362588,  0.8712592962592962, 0.8865485524576433, 0.8684139093230003, 0.7344155844155844, 0.7506493506493507] # node 1171
           ]
pcg = [[0.9880952380952382, 1.0, 0.9559824434824437, 0.9466931216931217, 0.8962962962962965, 0.8333333333333334],
        [0.8824645051917779, 0.8853275512366423, 0.8496821360457725, 0.8335664335664337,  0.7578512396694216,  0.806060606060606],
        [0.8562455221546131, 0.8712592962592962, 0.8579049738140647, 0.8634067952249771, 0.764141414141414, 0.712121212121212]]


colors = ['blue', 'green', 'red']
shades = ['lightblue', 'lightgreen', 'salmon']

# Create a scatter plot for blog
for i in range(len(blog)):
    plt.scatter(np.arange(1, len(blog_all[i])+1), blog_all[i], color=colors[i], label=f'node {i+1} all vote')
    plt.scatter(np.arange(1, len(blog[i])+1), blog[i], color=shades[i], label=f'node {i+1} train vote')

plt.xticks(np.arange(1,5))
plt.ylim(0.1, 0.5)

plt.legend()
# Save the plot
plt.savefig('blog_label_spread.png')

plt.clf()


# Create a line plot for blog
for i in range(len(blog)):
    plt.plot(np.arange(1, len(blog_all[i])+1), blog_all[i], color=colors[i], label=f'node {i+1} all vote')
    plt.plot(np.arange(1, len(blog[i])+1), blog[i], color=shades[i], label=f'node {i+1} train vote')

plt.xticks(np.arange(1,5))
plt.ylim(0.1, 0.5)

plt.legend()
# Save the plot
plt.savefig('blog_label_spread_line.png')

plt.clf()

# Creat a line plot for pcg
for i in range(len(pcg)):
    plt.scatter(np.arange(1, len(pcg_all[i])+1), pcg_all[i], color=colors[i], label=f'node {i+1} all vote')
    plt.scatter(np.arange(1, len(pcg[i])+1), pcg[i], color=shades[i], label=f'node {i+1} train vote')

plt.legend()
# Save the plot
plt.savefig('pcg_label_spread.png')


plt.clf()

# Creat a line plot for pcg
for i in range(len(pcg)):
    plt.plot(np.arange(1, len(pcg_all[i])+1), pcg_all[i], color=colors[i], label=f'node {i+1} all vote')
    plt.plot(np.arange(1, len(pcg[i])+1), pcg[i], color=shades[i], label=f'node {i+1} train vote')

plt.legend()
# Save the plot
plt.savefig('pcg_label_spread_line.png')
