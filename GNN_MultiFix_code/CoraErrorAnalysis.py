import re
from model import FPLPGCN, Local_Glbal_LC, FPLPGCN_dw, FPLPGCN_dw_linear
from data_loader import *
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter

if __name__ == "__main__":

    # Create and initialize the model
    G = load_cora()
    
    input_dim = G.x.shape[1]
    hidden_dim = 64
    output_dim = G.num_class

    model_names = ["FPLPGCN_linear_cora00___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora01___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora02___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora03___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora04___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora05___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora06___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora07___split_0.pt_checkpoint.pt",
                    "FPLPGCN_linear_cora08___split_0.pt_checkpoint.pt"]

    # settings
    settings = []

    # Open the file and find the lines that contain "ARGS"
    with open('sbatch_script/tune_scripts/cora_tune.sbatch', 'r') as file:
        for line in file:
            if "ARGS=" in line:
                # Find the numbers that come after "num_fp" and "num_lp"
                num_fp = re.search('--num_fp (\d+)', line)
                num_lp = re.search('--num_lp (\d+)', line)
                if num_fp and num_lp:
                    # Add the numbers to the list as integers
                    settings.append(int(num_fp.group(1)))
                    settings.append(int(num_lp.group(1)))

    settings = [settings[i:i+2] for i in range(0, len(settings), 2)]
    print(settings)

    scores = []            
    for i, model_name in enumerate(model_names):
        model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, 
                                  num_gcn_layers=settings[i][0], num_label_layers=settings[i][1])

        model.load_state_dict(torch.load(model_name))
        output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
        ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])
        scores.append(ap_test)
    print("the performances of the models: ", scores)

    # find the best model index
    best_model_index = max(enumerate(scores), key=lambda pair: pair[1])[0]
    print("the index of the best model: ", best_model_index)

    # load the model
    model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, 
                               num_gcn_layers=settings[best_model_index][0], 
                               num_label_layers=settings[best_model_index][1])

    model.load_state_dict(torch.load(model_names[best_model_index]))

    # the wrong predicted nodes
    out = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    _, pred = out.max(dim=1)
    wrong_indices = (pred != G.uncode_label).nonzero().flatten()

    wrong_indices_test = [i for i in wrong_indices if G.test_mask[i]]
    wrong_indices_test = torch.tensor(wrong_indices_test)

    # how many of the nodes are predicted wrong
    print("number of wrong predicted test nodes: ", wrong_indices_test.shape)
    print("number of test nodes: ", G.test_mask.sum().item())
    percentage_wrong_prediction = wrong_indices_test.shape[0] / G.test_mask.sum().item()
    print("percentage of nodes in the test data that are wrong: ", percentage_wrong_prediction)

    ############### from which classes are the nodes ##########################################
    wrong_test_nodes_labels = G.uncode_label[wrong_indices_test]
    wrong_test_nodes_labels_np = wrong_test_nodes_labels.numpy()
    plt.figure()
    plt.hist(wrong_test_nodes_labels_np, bins=range(torch.min(wrong_test_nodes_labels), torch.max(wrong_test_nodes_labels) + 2), align='left', rwidth=0.8)
    # Set the title and labels
    plt.xlabel('Label Index')
    plt.ylabel('Number of Nodes')
    plt.title('Histogram of Wrongly Predicted Node Classes')
    plt.savefig("wrong_label_distribution.png")

    ############## Plot the degree of the wrongly predicted nodes  ############################
    # how connected are the nodes
    adjacency_matrix = torch.zeros((G.num_nodes, G.num_nodes))
    adjacency_matrix[G.edge_index[0], G.edge_index[1]] = 1
    wrong_degrees = adjacency_matrix[wrong_indices_test].sum(dim=1).int().flatten().numpy()
    print("the degree of the wrong predicted nodes: ", wrong_degrees)
    bins = np.arange(wrong_degrees.min(), wrong_degrees.max() + 2) - 0.5
    plt.figure()
    plt.hist(wrong_degrees, bins=bins)
    # Set the title and labels
    plt.title('Histogram of Wrongly Predicted Nodes Degree')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.savefig("Wrong_degree_distribution.png")

    ############### Plot the percentage of wrongly predicted nodes ############################
    class_distribution = Counter(G.uncode_label.tolist())
    print('Class size distribution:', class_distribution)

    # check the percentage of wrong predicted nodes in each class
    frequency_wrong_class_test = dict(Counter(wrong_test_nodes_labels_np))
    result = [frequency_wrong_class_test[key] / class_distribution[key] for key in sorted(class_distribution.keys())]
    result = np.array(result)
    print("the len of the result: ", result.shape)
    print("result: ", result)
    plt.figure()

    plt.bar(np.arange(7), result)
    # Set the title and labels
    plt.title('Precentage of Wrong Predcted Nodes From Each Class')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.savefig("percentage_wrong_label_distribution.png")

    ############### check for isolated nodes ############################
    all_nodes = set(range(G.num_nodes))
    connected_nodes = set(G.edge_index[0].tolist() + G.edge_index[1].tolist())
    isolated_nodes = all_nodes - connected_nodes
    print('Number of isolated nodes:', len(isolated_nodes))



    # if a node is wrongly predicted, how many of its neighbors are wrongly predicted

    # homophily from the original true label and from the predicted labels

    # get insights from gcn also and compare











