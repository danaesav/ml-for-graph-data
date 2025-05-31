import numpy as np
import os
import scipy
import scipy.io
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
from gensim.models import Word2Vec, KeyedVectors
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def check_isolated_nodes(data):
    edge_index = data.edge_index
    all_nodes = torch.arange(data.y.shape[0])
    connected_nodes = torch.unique(edge_index)
    isolated_nodes = torch.tensor([node for node in all_nodes if node not in connected_nodes])
    return isolated_nodes

def pad_humloc(data_name="HumanGo", path=""):

    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (features.shape[0], features.shape[0]))

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.adj = adj

    num_nodes = labels.shape[0]
    G.n_id = num_nodes

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)
    print("isolated nodes type: ", isolated_nodes.type())

    if len(isolated_nodes) > 0:
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")


    # load deep walk embeddings, isolated nodes using random embeddings
    emb_model = KeyedVectors.load_word2vec_format("HumanGo/humloc.emb", binary=False)
    # Get the dimensions of the embeddings
    embedding_dimensions = emb_model.vector_size
    # Use random embeddings for the isolated nodes

    for node in list(isolated_nodes):
        random_embedding = np.random.rand(embedding_dimensions)
        emb_model.add_vector(str(int(node)), random_embedding)

    emb_model.save_word2vec_format('HumanGo/humloc_full.emb')


def pad_eukloc(data_name="EukaryoteGo", path=""):
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (labels.shape[0], labels.shape[0]))

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                            dtype=np.dtype(float), delimiter=',')).float()

    file_path = os.path.join(path, data_name, "split.pt")
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    G.adj = adj

    num_nodes = labels.shape[0]
    G.n_id = num_nodes

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix
    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)
    if len(isolated_nodes) > 0:
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")


    # load deep walk embeddings, isolated nodes using random embeddings
    emb_model = KeyedVectors.load_word2vec_format("EukaryoteGo/eukloc.emb", binary=False)
    # Get the dimensions of the embeddings
    embedding_dimensions = emb_model.vector_size

    # Use random embeddings for the isolated nodes
    for node in list(isolated_nodes):
        emb_model.add_vector(str(int(node)), np.random.rand(embedding_dimensions))

    emb_model.save_word2vec_format('EukaryoteGo/eukloc_full.emb')

# if not os.path.isfile("EukaryoteGo/eukloc_full.emb"):
#     pad_eukloc()
# if not os.path.isfile("HumanGo/humloc_full.emb"):
#     pad_humloc()


def check_hyper_data(data_name="Hyperspheres_10_10_0", split_name="split_0.pt", train_percent=0.6, path="", feature_noise_ratio=0.0):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = np.genfromtxt(osp.join(path, data_name, "features.csv"),
                             skip_header=1, dtype=np.dtype(float), delimiter=',')

    # remove the relevant features to the fixed relevant-irrelevant ratios
    start_col = int(10 * (1 - feature_noise_ratio))
    features = torch.tensor(features).float()[:, start_col:]

    edges = torch.tensor(np.genfromtxt(osp.join(path, data_name, "edges.txt"),
                         dtype=np.dtype(float), delimiter=',')).long()
    edge_index = torch.transpose(edges, 0, 1)

    # split
    folder_name = data_name + "_" + str(train_percent)
    file_path = osp.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    isolated_nodes = check_isolated_nodes(G)
    print("isolated nodes type: ", isolated_nodes.type())

    if len(isolated_nodes) > 0:
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")

    return G


# pad_eukloc()
# pad_humloc()
check_hyper_data()
