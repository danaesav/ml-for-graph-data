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
import argparse
import pandas as pd
from utils import compute_identity, compute_identity_sparse
from metrics import ap_score

############################################## Boxplot Loss Analyse GCN and Linear-GraphCAD ############################################# 

# Create the parser
parser = argparse.ArgumentParser(description='Process parameters')

# Add the arguments
parser.add_argument('--dir', type=str, default="",
                    #checkpoints_blogcatalog_split1_all_labels_GCN
                    #checkpoints_blogcatalog_split1_linear
                    #checkpoints_homo02_split0_clean_GCN
                    #checkpoints_homo02_split0_clean_linear
                    help='directry')
parser.add_argument('--data_name', type=str, help='data name')
parser.add_argument('--model_name', type=str, help='model name')

# Parse the arguments
args = parser.parse_args()

directory = args.dir
model_name = args.model_name
data_name = args.data_name
print("data name: ", data_name)
print("model name: ", model_name)
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


############################################# Visualize Losses Lineplot ################################################
def plot_losses(train_losses, test_losses, val_losses):
    # Convert the lists to tensors
    train_losses = torch.tensor(train_losses)
    test_losses = torch.tensor(test_losses)
    val_losses = torch.tensor(val_losses)
    print("train_losses: ", train_losses)
    print("test_losses: ", test_losses)
    print("val_losses: ", val_losses)


    avg_train_losses = torch.mean(train_losses, dim=1)
    std_train_losses = torch.std(train_losses, dim=1)
    avg_test_losses = torch.mean(test_losses, dim=1)
    std_test_losses = torch.std(test_losses, dim=1)
    avg_val_losses = torch.mean(val_losses, dim=1)
    std_val_losses = torch.std(val_losses, dim=1)

    ################################################# lineplot for all epochs ######################################

    # Plot the averages and fill between for the standard deviations
    # Create a figure and axis
    fig, ax = plt.subplots()
    if args.data_name == "blogcatalog":

        ax.plot(epochs, avg_train_losses, label='Train')
        ax.fill_between(epochs, avg_train_losses - std_train_losses, avg_train_losses + std_train_losses, alpha=0.5)

    else:
        # ax.plot(epochs, avg_train_losses, label='Train')
        # ax.fill_between(epochs, avg_train_losses - std_train_losses, avg_train_losses + std_train_losses, alpha=0.1)

        ax.plot(epochs, avg_test_losses, label='Test')
        ax.fill_between(epochs, avg_test_losses - std_test_losses, avg_test_losses + std_test_losses, alpha=0.5)

        # ax.plot(epochs, avg_val_losses, label='Validation')
        # ax.fill_between(epochs, avg_val_losses - std_val_losses, avg_val_losses + std_val_losses, alpha=0.1)

    # Set the labels
    ax.set_xlabel('Epoch', fontsize=18)
    ax.set_ylabel('Average Loss', fontsize=20)

    # Set xticks to every 6th number and the last element
    xticks = epochs[::6]
    if epochs[-1] not in xticks:
        xticks.append(epochs[-1])
    ax.set_xticks(xticks)
    ax.tick_params(axis='both', which='major', labelsize='x-large')

    # Set y-axis limits based on args.data_name
    if args.data_name == "blogcatalog":
        ax.set_ylim(0, 0.5)
    elif args.data_name == "dblp":
        ax.set_ylim(0, 0.9)

    # Add a legend
    ax.legend(fontsize=20)

    # Save the figure to a file
    plt.savefig("plot/lineplot_losses_"+args.model_name+"_"+args.data_name+".png")

    ############################################################################################################

    plt.clf()

    ################################# kde for last epoch ######################################################## 
    # Create a figure and axis
    fig, ax = plt.subplots()
    if args.data_name == "blogcatalog":
        print("last epoch", train_losses[-1, :])
        sns.kdeplot(train_losses[-1, :], label='Train', ax=ax,
                    fill=False,
                    common_norm=False, palette="crest", alpha=.5, linewidth=1.5,)

    else:

        print("last epoch", train_losses[-1, :])
        sns.kdeplot(train_losses[-1, :], label='Train', ax=ax,
                    fill=False,
                    common_norm=False, palette="crest", alpha=.5, linewidth=2,)
        sns.kdeplot(test_losses[-1, :], label='Test', ax=ax,
                    fill=False,
                    common_norm=False, palette="crest", alpha=.5, linewidth=2,)
        sns.kdeplot(val_losses[-1, :], label='Validation', ax=ax,
                    fill=False, common_norm=False, palette="crest", alpha=.5, linewidth=2,)
        
    plt.xscale('log')
    ax.tick_params(axis='both', which='major', labelsize='x-large')

    # Set the labels
    ax.set_xlabel('Loss', fontsize=18)
    ax.set_ylabel('Density', fontsize=20)

    ax.set_xlim(10**-2, 10**1)


    # Add a legend
    ax.legend(fontsize=20)
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("plot/kde_losses_"+args.model_name+"_"+args.data_name+".png")

    ############################################################################################################
####################################################################################################################


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


############################## Find Model Checkpoints #############################
file_names = get_file_names(directory)
file_names = sorted(file_names, key=extract_integer)
print(file_names)

# select the first 300 checkpoints
#file_names = [file for file in file_names if int(file.split('_')[1].split('.')[0]) < 300]

############### choose 30 epochs to visualize ##########
increment = len(file_names) // 30
complete_file_names = file_names
file_names = complete_file_names[::increment][:30]
print("the number of chosen epochs: ", len(file_names))

selected_indices = [complete_file_names.index(file) for file in file_names]
print("selcted indices: ", selected_indices)
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
        elif data_name=="dblp":
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=2, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
        else:
            model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=5, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
            
    model.load_state_dict(torch.load(directory+file_name))
    model.eval()
    models.append(model)
    print("model loaded")

################################################## Loss Calculation ##################################################
train_losses = []
test_losses = []
val_losses = []

# for each model calculate loss
for model in models:

    if model_name == "linear":
        output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    elif model_name == "GCN" or model_name == "IDGNN" or model_name=="H2GCN":
        output = model(G.x, G.edge_index)

    # read the output for train and test
    train_output = output[G.train_mask]
    test_output = output[G.test_mask]
    val_output = output[G.val_mask]

    # evaluation
    ap_train = ap_score(G.y[G.train_mask], train_output)
    ap_test = ap_score(G.y[G.test_mask], test_output)
    ap_val = ap_score(G.y[G.val_mask], val_output)
    print("ap train: ", ap_train, "ap_test", ap_test, "ap_val", ap_val)

    # ground truth for train and test
    train_true = G.y[G.train_mask]
    test_true = G.y[G.test_mask]
    val_true = G.y[G.val_mask]

    train_loss = []
    # losses on the training nodes
    for i, pred in enumerate(train_output):
        loss = BCE_loss(pred, train_true[i])
        train_loss.append(loss.item())

    test_loss = []
    # losses on the test nodes
    for j, pred in enumerate(test_output):
        loss = BCE_loss(pred, test_true[j])
        test_loss.append(loss.item())

    val_loss = []
    # losses on the test nodes
    for k, pred in enumerate(val_output):
        loss = BCE_loss(pred, val_true[k])
        val_loss.append(loss.item())
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    val_losses.append(val_loss)

# Plot losses
plot_losses(train_losses, test_losses, val_losses)





