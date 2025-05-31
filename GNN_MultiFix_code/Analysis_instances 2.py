import torch
import numpy as np
from scipy import stats
from args import get_args
import torch.nn.functional as F
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
from utils import row_normalize
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
import torch_geometric.utils as utils
from collections import Counter
import networkx as nx
import numpy as np

def get_hop_adjacency_matrix(G, node, hop):
    # Convert PyTorch Geometric graph to NetworkX graph
    G_nx = utils.to_networkx(G)

    # Initialize an adjacency matrix with zeros
    adj_matrix = np.zeros((G.num_nodes, G.num_nodes))

    # Get the neighbors at the specific hop
    neighbors = nx.single_source_shortest_path_length(G_nx, node, cutoff=hop)

    # Exclude neighbors from previous hops
    neighbors = {k: v for k, v in neighbors.items() if v == hop}

    # Update the adjacency matrix
    for neighbor in neighbors:
        for other in G_nx[neighbor]:
            if other in neighbors:
                adj_matrix[neighbor][other] = 1

    return adj_matrix


if __name__ == "__main__":

    ######## Hyperparameter Setting #########
    args = get_args()
    args.data_name = "yelp"
    print("########Hyperparameter setting########")
    print("data_name: ", args.data_name)
    print("train percent: ", args.train_percent)
    print("Split name: ", args.split_name)
    print("#######################################")
    ######## import data #########
    if args.data_name == "pcg":
        G = load_pcg(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "humloc":
        G = load_humloc()

    elif args.data_name == "eukloc":
        G = load_eukloc()

    elif args.data_name == "yelp":
        G = load_yelp(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "blogcatalog":
        G = load_blogcatalog(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "hyper":
        print("training percent: ", args.train_percent)
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                            feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)
        print("training size: ", G.train_mask.sum())

    elif args.data_name == "dblp":
        G = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "cora":
        G = load_cora()

    elif args.data_name == "citeseer":
        G = load_citeseer()

    # multi-class
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        output_dim = G.num_class
        print("this is a multi-class dataset, output dim is number of classes")
        args.multi_class = True

    # multi-label
    else:
        output_dim = G.y.shape[1]
        args.multi_class = False


    # Get the indices of the test nodes
    test_indices = G.test_mask.nonzero(as_tuple=True)[0]
    
    ##################### multi-class dataset #############################
    if args.multi_class:
        #
        print("to be implemented for multi-class daatset")
        
          
    ##################### multi-label dataset #############################
    else:
        ########################### High Label Assignment Nodes ######################

        label_counts = Counter(G.y.tolist())
        print("label counts: ", label_counts)
        # Get the 3 most common labels
        top3_labels = [label for label, count in label_counts.most_common(3)]

        # Get the indices of the nodes with the top 3 labels
        top3_nodes = [index for index, label in enumerate(labels.tolist()) if label in top3_labels]
        print(top3_nodes)

        hops = [1, 2, 10, 15, 20]
        for node in top3_nodes:
            for hop in hops:
                adj_matrix = get_hop_adjacency_matrix(G, node, hop)
                print(f"Adjacency matrix for {hop} hops away:")
                print(adj_matrix)




        