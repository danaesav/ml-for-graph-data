import torch
import matplotlib.pyplot as plt
from model import FPLPGCN_dw_linear
from baseline_models import GCN, H2GCN
from data_loader import load_pcg, load_blogcatalog, load_hyper_data, load_DBLP
import os
from metrics import BCE_loss
import re
import numpy as np
import seaborn as sns
from torch_geometric.utils import degree
import pandas as pd
import argparse
import dgl
from utils import compute_identity, compute_identity_sparse
from sklearn.metrics import jaccard_score
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, is_undirected
from metrics import f1_loss, BCE_loss, rocauc_, ap_score


############################################## Observe Hard Nodes Characteristics ############################################# 
#torch.set_printoptions(edgeitems=3, threshold=10000, linewidth=200, precision=2)

# Create the parser
parser = argparse.ArgumentParser(description='Process parameters')

# Add the arguments
parser.add_argument('--dir', type=str, default="",
                    #specified in the sbatch file
                    help='directry')
parser.add_argument('--data_name', type=str, help='data name')
parser.add_argument('--model_name', type=str, help='model name')

# Parse the arguments
args = parser.parse_args()

directory = args.dir
model_name = args.model_name
data_name = args.data_name
##########################################################################################################################################     

def get_file_names(directory):
    return os.listdir(directory)

def extract_integer(s):
    match = re.search(r'_([0-9]+)\.', s)
    if match:
        return int(match.group(1))
    else:
        return None
   

def get_checkpoints(directory, pattern):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.startswith(pattern)]
    return files


# for each label, save the indices of the nodes that belong to this class
def label_assign(G):
    # key: each class index
    # values: indices of nodes that have this class label
    labels = G.y

    class_node = {}
    classes_tensor = torch.transpose(labels, 0, 1)

    for index, row in enumerate(classes_tensor):
        class_node[index] = torch.flatten(torch.nonzero(row))

    return class_node

# the label dist in the neighborhood of each node and each label
def label_dist(data):
    edge_index = data.edge_index
    y = data.y

    class_asgn = label_assign(data)

    # Ensure the graph is undirected
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)

    # Get the number of nodes and labels
    num_nodes = data.num_nodes
    num_labels = y.shape[1]

    # Initialize the label distribution matrix for each node in its neighborhood
    label_distribution = torch.zeros(num_nodes, num_labels, device=y.device)

    # Calculate the label distribution in the neighborhood of each node
    for node in range(num_nodes):
        # node = 235
        neighbors = edge_index[1, edge_index[0] == node]
        neighbor_labels = y[neighbors]
        #print("neighbor_labels: ", neighbor_labels)
        column_sums = torch.sum(neighbor_labels, dim=0)
        #print("col sum:", column_sums)
        total_sum = torch.sum(column_sums)
        #print("total sum: ", total_sum)
        label_distribution[node] = column_sums / total_sum
        #print("label dist", label_distribution[node])
        #break

    # summarize the label distribution in the neighborhood of each label in the graph
    norm_label_dist_per_label = {}
    for label in range(y.size(1)):
        node_indices = class_asgn[label]
        # get their neighbors
        neighbor_indices = []
        for n in node_indices:
            neighbors = edge_index[1, edge_index[0] == n]
            neighbor_indices.append(neighbors)
        # maybe duplicates inds in it
        neighbor_indices = torch.tensor([item for sublist in neighbor_indices for item in sublist])
        distributions = y[neighbor_indices]
        # Sum over each column
        column_sums = torch.sum(distributions, dim=0)
        total_sum = torch.sum(column_sums)
        # Normalize the column sums
        normalized_column_sums = column_sums / total_sum

        norm_label_dist_per_label[label] = normalized_column_sums
        
    return label_distribution, norm_label_dist_per_label


############################################# Visualize Losses ################################################
def hard_nodes_analysis(train_losses, test_losses):
    print(torch.tensor(train_losses).shape) # epochs, num_train_nodes
    print(torch.tensor(test_losses).shape) # epochs, num_test_nodes
    
    # Convert the tensor to a DataFrame
    train_losses = np.array(train_losses) # epochs, num_train_nodes
    df = pd.DataFrame(train_losses)
    print("df dim: ", df.shape)
    print(df)
    ########### get the outliers of the box plot #######


    # Find the indices of the outliers
    # outliers_dict = {}
    # for epoch in range(df.shape[0]):
    #     # analyse only the last epoch 
    #     epoch = df.shape[0] - 1
    #     Q1 = df.iloc[epoch].quantile(0.25)
    #     Q3 = df.iloc[epoch].quantile(0.75)
    #     IQR = Q3 - Q1
    #     outliers = (df.iloc[epoch] > (Q3 + 1.5 * IQR))  & (df.iloc[epoch] > 2) # boolean mask
    #     outliers_dict[epoch] = np.where(outliers)[0].tolist()
    #     break

    ################### find outlier in the last epoch ########################
    outliers_dict = {}
    epoch = df.shape[0] - 1
    Q1 = df.iloc[epoch].quantile(0.25)
    Q3 = df.iloc[epoch].quantile(0.75)
    IQR = Q3 - Q1
    outliers = (df.iloc[epoch] > (Q3 + 1.5 * IQR)) & (df.iloc[epoch] > 2)  # boolean mask
    outliers_dict[epoch] = np.where(outliers)[0].tolist()
    ####################################################################################
       
    print("the chosen indices on the outliers: ", outliers_dict)

    # key: epoch # value: indices of the nodes
    for key, value in outliers_dict.items():
        # Print the length of each value
        print(f'Number of outliers in epoch {key}: {len(value)}')

    ########## check outliers homophily and ccns score ############
    outliers_per_epoch = outliers_dict.values() # outlier indices
    all_outlier_indices = [item for sublist in outliers_per_epoch for item in sublist]
    # indices among traning nodes
    outliers_indices = list(set(all_outlier_indices))
    outliers_indices.sort()
    print("check outlier_indices among training nodes:", len(outliers_indices), outliers_indices)

    # the indices of the nodes in the whole graph
    train_indices = torch.where(G.train_mask)[0]
    print("number of training nodes: ", train_indices.shape)
    outliers_indices_in_all = train_indices[outliers_indices]

    # losses of the outlier in last checkpoint
    training_losses_last_checkpoint = df.iloc[-1].values
    print("training_losses_last_checkpoint shape: ", training_losses_last_checkpoint.shape)

    training_losses_last_checkpoint_outlier = training_losses_last_checkpoint[outliers_indices]
    print("losses last checkpoint: ", training_losses_last_checkpoint_outlier)
    homos_outlier = []
    degrees_outlier = []

    for o, node_index in enumerate(outliers_indices_in_all):
        print("####################################")
        print("node index: ", node_index)
        # label vector of the node
        y_target = G.y[node_index]
        # label indices of the node
        lbl_target = y_target.nonzero(as_tuple=True)[0].tolist()
        print("label vector of target node:", y_target)
        print("label set of target node:", lbl_target)

        # the neighbor indices
        neighbor_indices = G.edge_index[1, G.edge_index[0] == node_index]
        print("neighbors of target node: ", neighbor_indices)
        degrees_outlier.append(neighbor_indices.shape[0])
        # Get the labels of its neighbors
        neighbor_labels = G.y[neighbor_indices]
        print("labels of neighbors",neighbor_labels)
        
        # if the node has no neighbor
        if neighbor_indices.numel() == 0:
            homo_instance = 0.0
            ccns_score = 1.0
            print("isolated node.")
            print("instane level homophily: ", homo_instance)
            print("ccns score", ccns_score)
            homos_outlier.append(homo_instance)
        # the node has at least one neighbor
        else:

            # instance-homo
            homo_instance = 0.0
            for neigh_id in torch.arange(neighbor_labels.shape[0]):
                print("similary with one neighbor: ", jaccard_score(neighbor_labels[neigh_id], y_target, average='binary'))
                homo_instance += jaccard_score(neighbor_labels[neigh_id], y_target, average='binary')
            homo_instance = homo_instance / neighbor_labels.shape[0]
            if homo_instance == 1.0:
                print("homo=1 alert!!")
                print("loss on the node: ", training_losses_last_checkpoint_outlier[o])
            homos_outlier.append(homo_instance)
                
            #print("homophily hard score: ", 1 - homo_instance)

            # ccns
            ccns_score = 0
            smoothing_factor=1e-10
            # the label dist in the neighborhood of target_node
            for l in lbl_target:
                print("lbl_dist_per_labe: ", G.lbl_dist_per_label[l])
                kl_div = F.kl_div(torch.log(G.lbl_dist_per_node[node_index]+smoothing_factor), G.lbl_dist_per_label[l], reduction='batchmean')
                ccns_score += kl_div
            ccns_score = ccns_score / len(lbl_target)
            print("ccns score", ccns_score)
    
    # plot loss with homo and degree
    # Create a scatter plot
    plt.scatter(homos_outlier, training_losses_last_checkpoint_outlier)

    # Set the title and labels
    plt.title('Scatter plot of training losses and homos outlier')
    plt.ylabel('Training losses')
    plt.xlabel('Instance Label Homophily Of Outliers')

    # Save the figure
    plt.savefig('../Data_Hardness/training_loss_homo.png')

    # Clean the previous plot
    plt.clf()

    # Create a scatter plot
    plt.scatter(degrees_outlier, training_losses_last_checkpoint_outlier)

    # Set the title and labels
    plt.title('Scatter plot of training losses and degree outlier')
    plt.ylabel('Training losses')
    plt.xlabel('Degree Of Outliers')

    # Save the figure
    plt.savefig('../Data_Hardness/training_loss_degree.png')

            

###########################################################################################################


########################################### Load Data ###################################################
if data_name == "blogcatalog":
    G = load_blogcatalog(split_name="split_1.pt", train_percent=0.6)
    # set all nodes for training in baselines
    if model_name != "linear":
        print("check train mask before:", G.train_mask)
        train_mask = torch.ones(G.x.shape[0]).bool()
        G.train_mask = train_mask
        G.val_mask = train_mask
        print("check train mask after, all nodes to be true:", G.train_mask)
elif data_name == "hyper":
    G = load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=None, homo_level="homo02")

elif data_name == "dblp":
    G = load_DBLP(split_name="split_1.pt", train_percent=0.6)

    ####################### check for duplicate ######################




# add analysis needed characteristics
G.lbl_dist_per_node, G.lbl_dist_per_label  = label_dist(G)

# PREPROCESS FOR IDGNN
# preprocessing fro IDGNN, calculate the indenty info
if args.model_name == "IDGNN":
    if args.data_name == "yelp":
        # not possible to use adj_dense
        identity = compute_identity_sparse(G.edge_index, n=G.x.shape[0], k=3)
    else:
        # use adj_dense
        identity = compute_identity(G.edge_index, n=G.x.shape[0], k=3) # (n, k)
    print("identity dim: ", identity.shape)
    # inject the identity info into feat
    G.x = torch.cat((G.x, identity), dim=1)
    print("G.x SHAPE", G.x.shape)
###########################################################################################################


############################################ Find Model Checkpoints #########################################
file_names = get_file_names(directory)
file_names = sorted(file_names, key=extract_integer)
print(file_names)

# select the first 300 checkpoints
file_names = [file for file in file_names if int(file.split('_')[1].split('.')[0]) < 110]

############### choose 30 epochs ##########
increment = len(file_names) // 30
complete_file_names = file_names
file_names = complete_file_names[::increment][:30]
print("the number of chosen epochs: ", len(file_names))

selected_indices = [complete_file_names.index(file) for file in file_names]
print("selcted file indices in the file list: ", selected_indices)
selected_file_names = [complete_file_names[i] for i in selected_indices]
print("selecetd file names: ", selected_file_names)
epochs = [int(re.search(r'_(\d+)\.', name).group(1)) for name in selected_file_names]
print("chosen epochs: ", epochs)
# initialization
input_dim = G.x.shape[1]
output_dim = G.y.shape[1]
hidden_dim = 256

################################################### load the models ##################################################
models = []

for file_name in file_names:
    if model_name=="GCN":
        model = GCN(in_channels=input_dim, hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=False)

    elif args.model_name.startswith("IDGNN"):
        # use normal GCN WITH AUGMENTED FEATURES
        model = GCN(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=False)

    elif args.model_name.startswith("H2GCN"):
        model = H2GCN(nfeat=G.x.shape[1], nhid=hidden_dim, nclass=G.y.shape[1], dropout=0.5, multi_class=False)
    else: 
        if data_name=="blogcatalog":
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
        else:
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=5, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
    model.load_state_dict(torch.load(directory+file_name))
    model.eval()
    models.append(model)

################################################## Loss Calculation ##################################################
train_losses = []
test_losses = []
# for each model calculate loss
for model in models:

    if model_name == "linear":
        output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    elif model_name == "GCN" or model_name == "IDGNN" or model_name=="H2GCN":
        output = model(G.x, G.edge_index)

    # read the output for train and test
    train_output = output[G.train_mask]
    test_output = output[G.test_mask]
    print("train output: ", train_output)

    # evaluation
    ap_train = ap_score(G.y[G.train_mask], train_output)
    ap_test = ap_score(G.y[G.test_mask], test_output)
    print("ap train: ", ap_train, "ap_test", ap_test)

    # ground truth for train and test
    train_true = G.y[G.train_mask]
    test_true = G.y[G.test_mask]

    train_loss = []
    # losses on the training nodes
    for i, pred in enumerate(train_output):
        loss = BCE_loss(pred, train_true[i])
        train_loss.append(loss.item())
    #print("train losses: ", train_loss)

    test_loss = []
    # losses on the test nodes
    for j, pred in enumerate(test_output):
        loss = BCE_loss(pred, test_true[j])
        test_loss.append(loss.item())
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# analyse hard nodes
hard_nodes_analysis(train_losses, test_losses)





