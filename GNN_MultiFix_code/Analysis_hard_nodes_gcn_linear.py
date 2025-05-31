import seaborn as sns
from torch_geometric.utils import degree
from data_loader import load_pcg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

# Load data
G = load_pcg(split_name="split_0.pt", train_percent=0.6)

gcn_hard = [1350, 1567,  610,  565,  178,  129,  521,  599, 680, 1010, 2752,  564,  374,  889,
  701, 1481, 1554, 1906, 1855, 1207, 257, 109, 1573, 963, 177, 682, 14, 1843
, 1557, 422, 1394, 702, 1617, 522, 1119, 1042, 15, 1171, 206, 785, 1392, 3
, 583, 1851, 1480, 379, 1942, 1050, 1391, 451, 1545, 184, 850, 1300, 99, 952
, 649, 1439, 853, 3062, 1484, 1890, 1347, 78, 955, 2414, 858, 158, 224, 1734
, 2033, 1618, 386, 2073, 2112, 1581, 783, 1120, 1781, 1063, 544, 1002, 1017, 2388
, 1744, 1973, 390, 1946, 417, 1561, 512, 1167, 1537, 1348, 1584, 380, 609, 1880
, 61, 1778]
linear_hard =  [ 610, 433, 932, 177, 970, 522, 1906,  702, 1761, 1042,  109,  119, 1573, 1557
, 1579,  682, 1394, 1119, 1417, 1617, 2960, 1396,  785,  773, 1449, 1207,  680, 1272
, 1350,  178, 1843, 1481,  889,  206, 1392,  379,  963, 3,  257,  701, 1171, 1942
,  564, 1050,  15, 1391, 1480, 1851,  451,  583, 2503,  850,  390, 1270, 1365, 1581
, 2334, 2073, 1744, 3062,  594,  97, 1890, 2726, 1439,  903,  309,  783,  78, 1973
, 1545,  417,  386, 1063,  544, 1618, 1002,  853, 1484, 1300, 1781, 2112, 2414,  858
, 1348,  99, 1347, 1584,  955,  649, 1537,  380, 1120, 1167, 1017,  512,  609,  61
, 1880, 1778]


# Get the elements that are in both lists
common_elements = list(set(gcn_hard) & set(linear_hard))

# only_gcn_hard = list(set(gcn_hard) - set(common_elements))

# only_linear_hard = list(set(linear_hard) - set(common_elements))

only_gcn_hard = gcn_hard

only_linear_hard = linear_hard

print("############## Hard for GCN ####################")
degrees_gcn_hard = []
num_labels_gcn_hard = []
for node_index in only_gcn_hard:
        # degree
        node_degree = degree(G.edge_index[0])[node_index]
        # num of labels
        node_labels = G.y[node_index]
        num_labels = node_labels.sum().item()

        print(f'Degree of node {node_index}: {node_degree}')
        print(f'Number of labels of node {node_index}: {num_labels}')

        degrees_gcn_hard.append(node_degree)
        num_labels_gcn_hard.append(num_labels)

        print("#######################################################")


print("############## Hard for Linear ####################")
degrees_linear_hard = []
num_labels_linear_hard = []
for node_index in only_linear_hard:
        # degree
        node_degree = degree(G.edge_index[0])[node_index]
        # num of labels
        node_labels = G.y[node_index]
        num_labels = node_labels.sum().item()

        print(f'Degree of node {node_index}: {node_degree}')
        print(f'Number of labels of node {node_index}: {num_labels}')
        degrees_linear_hard.append(node_degree)
        num_labels_linear_hard.append(num_labels)

        print("#######################################################")


############# Visualize Dist ##################
colors = sns.color_palette()
# Create a DataFrame from the two lists
df = pd.DataFrame({'A': np.array(degrees_gcn_hard), 'B': np.array(degrees_linear_hard)})
plt.figure()
sns.histplot(df['A'], color=colors[0], alpha=0.5)
sns.histplot(df['B'], color=colors[1], alpha=0.5)
plt.xlabel("Degree Of The Node")

# Create custom patches
patch1 = mpatches.Patch(color=colors[0], label='Hard For GCN')
patch2 = mpatches.Patch(color=colors[1], label='Hard For GraphCAD')
plt.legend(handles=[patch1, patch2])
plt.savefig("Ablation_degree_hard_linear_gcn.png")

# Create a DataFrame from the two lists
df1 = pd.DataFrame({'A': np.array(num_labels_gcn_hard), 'B': np.array(num_labels_linear_hard)})
plt.figure()  # Create a new figure
sns.histplot(df1['A'], color=colors[0], alpha=0.5)
sns.histplot(df1['B'], color=colors[1],alpha=0.5)
plt.xlabel("Number Of Label Per Node")


patch1 = mpatches.Patch(color=colors[0], label='Hard For GCN')
patch2 = mpatches.Patch(color=colors[1], label='Hard For GraphCAD')
plt.legend(handles=[patch1, patch2])

plt.savefig("Ablation_num_labels_hard_linear_gcn.png")