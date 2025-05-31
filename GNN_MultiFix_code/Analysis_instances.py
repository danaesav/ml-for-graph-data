import torch
import numpy as np
from scipy import stats
from args import get_args
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from utils import row_normalize
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
import torch_geometric.utils as utils
from collections import Counter
import networkx as nx
import numpy as np
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import jaccard_score

def get_hop_neighbors(data, node, hop):
    # Convert edge_index to dense adjacency matrix
    adj_matrix = to_dense_adj(data.edge_index)[0]

    # Get neighbors at each hop up to the specified hop
    neighbors_at_each_hop = []
    power_adj_matrix = torch.eye(adj_matrix.size(0))
    for _ in range(hop):
        power_adj_matrix = torch.matmul(power_adj_matrix, adj_matrix)
        neighbors = (power_adj_matrix[node] > 0).nonzero(as_tuple=True)[0].tolist()
        neighbors_at_each_hop.append(neighbors)

    # Exclude neighbors from lower hops
    hop_neighbors_excluding_lower_hops = [neighbor for neighbor in neighbors_at_each_hop[-1] if neighbor not in [item for sublist in neighbors_at_each_hop[:-1] for item in sublist]]

    return hop_neighbors_excluding_lower_hops


if __name__ == "__main__":

    ######## Hyperparameter Setting #########
    args = get_args()

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
        # For each test node, get the labels of its neighbors and take the majority vote
        print("to be implemented for multi-class daatset")
        
          
    ##################### multi-label dataset #############################
    else:

        ########################################## High Label Assignment Nodes ###########################################
        # get the most labeled nodes in the graph
        counts = torch.count_nonzero(G.y, dim=1)
        label_counts, node_indices = torch.topk(counts, 3)
        print("Node indices: ", node_indices)
        print("Number of labels: ", label_counts)

        # analysis different hops of neighbors
        hops = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # analyse homophily
        for node in node_indices:

            print("####################################")
            print("analysis node with index: ", node)

            for hop in hops:
                print(hop, "hop away neighborhood")
                neighbors = get_hop_neighbors(G, node, hop)

                if len(neighbors) >= 1 :
                    print("number of neighbors in this hop: ", len(neighbors))

                    ############################# Hops Away Homophily Analysis ###############################
                    # calculate the homophily in this hop
                    #support = 0.0
                    #for n in neighbors:
                        #support = support + jaccard_score(G.y[node], G.y[n])
                    #h = support / len(neighbors)
                    #print("homophily level: ", h)

                    ###########################################################################################
                    

                    ############################# Majority Vote Analysis ######################################
                    # get the neighbors
                    neighbor_indices = neighbors
                    # get the label matrix of the neighbors
                    neighbor_labels = G.y[neighbor_indices]

                    # zero padding
                    for k, n in enumerate(neighbor_indices):
                        if G.val_mask[n] or G.test_mask[n]:
                            neighbor_labels[k] = torch.zeros(G.y.shape[1])

                    # sum to get the label distribution in the neighborhood
                    neighbor_labels_sum = neighbor_labels.sum(dim=0)
                    # normalize the sum to get propabilities
                    normalized_sum = neighbor_labels_sum / neighbor_labels_sum.sum()

                    # none of the neighbors are labeled, then fill the vote with zeros
                    if neighbor_labels_sum.sum()==0:
                        normalized_sum = torch.zeros(G.y.shape[1])
                    
                    print(G.y[node].shape)
                    print(normalized_sum.shape)
                    ap_score_test = average_precision_score(G.y[node], normalized_sum)
                    #micro_test, macro_test = f1_loss(G.y[node], normalized_sum)
                    print("Evaluation: the majority vote in ", hop, "away neighborhood for node", node)
                    #print("Micro and Macro F1 score: ", micro_test, macro_test)
                    print("Average precision:", ap_score_test)
                   
                    
                    ############################################################################################
                else:
                    print("Alll neighbors in this hop already included in lower hops")
                    print("No need to go to higher order")
                    print("try next node")
                    break



        