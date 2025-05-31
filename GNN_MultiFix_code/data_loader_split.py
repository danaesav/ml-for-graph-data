from metrics import *
import numpy as np
import os
import scipy
import scipy.io
import os.path as osp
from torch_geometric.data import Data
from torch_geometric.datasets import Yelp
from utils import sparse_mx_to_torch_sparse_tensor, row_normalize, remove_and_reindex_isolated_nodes, check_isolated_nodes
from gensim.models import Word2Vec, KeyedVectors
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F


def load_pcg(data_name="pcg_removed_isolated_nodes", split_name="split_0.pt", train_percent=0.6, path="data/"):

    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                                          dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()

    # splits is a nested dictionary
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
             y=labels.float())
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1/G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/pcg_removed_isolated_nodes/pcg.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()


    ################################# Split train to Supervision + Train ###################
    supervision_split_file_name = os.path.join(path, folder_name, split_name[:-3] + "_supervision_split.pt")
    print("supervision_split_file_name: ", supervision_split_file_name)
    # Check if supervision mask already exist
    if os.path.isfile(supervision_split_file_name):
        print("loading the supervision split...")
        # If the file exists, load the supervision mask from the file
        supervision_mask = torch.load(supervision_split_file_name)
        G.supervision_mask = supervision_mask
        # update the rest of the train node as input for label propagation
        G.label_input_mask = G.train_mask & ~G.supervision_mask

    # split the training nodes
    else:
        print("spliting the training nodes into supervision and label input...")
        # Get indices of training nodes
        train_indices = np.where(G.train_mask)[0]

        # Randomly split the training nodes into two groups
        np.random.shuffle(train_indices)
        split_index = len(train_indices) // 2

        # Create supervision_mask and update train_mask
        G.supervision_mask = np.zeros_like(G.train_mask, dtype=bool)
        G.supervision_mask[train_indices[:split_index]] = True
        # Convert the numpy arrays to PyTorch tensors
        G.supervision_mask = torch.from_numpy(G.supervision_mask)
        # Create label_input_mask as the difference between train_mask and supervision_mask
        G.label_input_mask = G.train_mask & ~G.supervision_mask

        torch.save(G.supervision_mask, supervision_split_file_name)


    # Check if the supervision_mask plus label_input_mask covers all the train_mask
    assert torch.all((G.supervision_mask | G.label_input_mask) == G.train_mask), "The supervision_mask plus label_input_mask do not cover all the train_mask."
    print("number of supervision nodes", torch.sum(G.supervision_mask).item())
    print("number of train nodes", torch.sum(G.train_mask).item())
    print("number of all nodes: ", G.x.shape[0])
    print("number of supervision nodes", torch.sum(G.label_input_mask).item())
    print("check unequal: ", int(torch.sum(G.train_mask).item() / 2), torch.sum(G.train_mask).item() / 2)


    # Check if the supervision_mask and label_input_mask are each of 50% size of the training nodes
    assert torch.sum(G.supervision_mask).item() == int(torch.sum(G.train_mask).item() / 2), "The supervision_mask is not 50% of the training nodes."

    # padd the input label matrix for supervision nodes
    G.y_pad[G.supervision_mask] = padding

    print("check the label input for supervision nodes: ", G.y_pad[G.supervision_mask])
    print("check the label input for label input nodes: ", G.y_pad[G.label_input_mask])

    return G


def load_hyper_data(data_name="Hyperspheres_10_10_0", split_name="split_0.pt", train_percent=0.6, path="data/", 
                    feature_noise_ratio=None, homo_level=None):
    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    # load complete features
    features = np.genfromtxt(osp.join(path, data_name, "features.csv"),
                             skip_header=1, dtype=np.dtype(float), delimiter=',')

    
    ######################## Feature Noise Experiments ########################
    # remove the relevant features to the fixed relevant-irrelevant ratios
    if (feature_noise_ratio != None) &  (homo_level == None):
        start_col = int(10.0 * (round(1.0 - feature_noise_ratio, 1)))
        features = torch.tensor(features).float()[:, start_col:]
        print("removing the relevant features...")
        print("input feature dim: ", features.shape[1])

        print("loading edges in homo02 for feature quality experiment...")
        # edges = torch.tensor(np.genfromtxt(osp.join(path, data_name, "edges.txt"),
        #                      dtype=np.float32, delimiter=',')).long()
        # edge_index = torch.transpose(edges, 0, 1)
        edges = torch.tensor(np.genfromtxt(osp.join(path, data_name, "homo02.txt"),
                             dtype=np.float32, delimiter=',')).long()
        edge_index = torch.transpose(edges, 0, 1)
        
    ######################## Homophily Level Experiment #######################
    # load corresponding edges
    if (feature_noise_ratio == None) &  (homo_level != None):
        ############### Only Relevant Features ##############
        features = torch.tensor(features).float()[:, :10]
        print("feature dim: ", features.shape[1])
        edges = torch.tensor(np.genfromtxt(osp.join(path, data_name, homo_level+".txt"),
                             dtype=np.float32, delimiter=',')).long()
        edge_index = torch.transpose(edges, 0, 1)
        print("loading the edges for homophily level...")
        print(edge_index.shape)


    # load split
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
    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix


    #####################################################################################
    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    #####################################################################################
    # if (feature_noise_ratio != None) &  (homo_level == None):
    #     emb_model = KeyedVectors.load_word2vec_format("data/Hyperspheres_10_10_0/hypersphere.emb", binary=False)
    #     deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    #     G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()
    ############################################################################

    # load deep walk embeddings
    # remove the relevant features to the fixed relevant-irrelevant ratios
    if (feature_noise_ratio != None) &  (homo_level == None):
        emb_model = KeyedVectors.load_word2vec_format("data/Hyperspheres_10_10_0/hyper02.emb", binary=False)
        deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    # for homophily level experiment load the corresponding embs
    if (feature_noise_ratio == None) &  (homo_level != None):
        print("loading the emb for: ", "data/Hyperspheres_10_10_0/homo"+homo_level[-2:]+".emb")
        # load emb model
        emb_model = KeyedVectors.load_word2vec_format("data/Hyperspheres_10_10_0/hyper"+homo_level[-2:]+".emb", binary=False)
        # random initialize the emb matrix
        deep_walk_emb = np.random.randn(labels.shape[0], 64)
        # replace if there is emb: for isolated nodes with no emb, random initialize
        for node in range(G.num_nodes):
            if str(node) in emb_model.index_to_key:
                deep_walk_emb[node] = np.asarray([emb_model[str(node)]])
        
        print("deep walk_emb shape: ", deep_walk_emb.shape)
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    ################################# Split train to Supervision + Train ###################
    supervision_split_file_name = os.path.join(path, folder_name, split_name[:-3] + "_supervision_split.pt")
    print("supervision_split_file_name: ", supervision_split_file_name)
    # Check if supervision mask already exist
    if os.path.isfile(supervision_split_file_name):
        print("loading the supervision split...")
        # If the file exists, load the supervision mask from the file
        supervision_mask = torch.load(supervision_split_file_name)
        G.supervision_mask = supervision_mask
        # update the rest of the train node as input for label propagation
        G.label_input_mask = G.train_mask & ~G.supervision_mask

    # split the training nodes
    else:
        print("spliting the training nodes into supervision and label input...")
        # Get indices of training nodes
        train_indices = np.where(G.train_mask)[0]

        # Randomly split the training nodes into two groups
        np.random.shuffle(train_indices)
        split_index = len(train_indices) // 2

        # Create supervision_mask and update train_mask
        G.supervision_mask = np.zeros_like(G.train_mask, dtype=bool)
        G.supervision_mask[train_indices[:split_index]] = True
        # Convert the numpy arrays to PyTorch tensors
        G.supervision_mask = torch.from_numpy(G.supervision_mask)
        # Create label_input_mask as the difference between train_mask and supervision_mask
        G.label_input_mask = G.train_mask & ~G.supervision_mask

        torch.save(G.supervision_mask, supervision_split_file_name)


    # Check if the supervision_mask plus label_input_mask covers all the train_mask
    assert torch.all((G.supervision_mask | G.label_input_mask) == G.train_mask), "The supervision_mask plus label_input_mask do not cover all the train_mask."
    print("number of supervision nodes", torch.sum(G.supervision_mask).item())
    print("number of train nodes", torch.sum(G.train_mask).item())
    print("number of all nodes: ", G.x.shape[0])
    print("number of supervision nodes", torch.sum(G.label_input_mask).item())
    print("check unequal: ", int(torch.sum(G.train_mask).item() / 2), torch.sum(G.train_mask).item() / 2)


    # Check if the supervision_mask and label_input_mask are each of 50% size of the training nodes
    assert torch.sum(G.supervision_mask).item() == int(torch.sum(G.train_mask).item() / 2), "The supervision_mask is not 50% of the training nodes."

    # padd the input label matrix for supervision nodes
    G.y_pad[G.supervision_mask] = padding

    print("check the label input for supervision nodes: ", G.y_pad[G.supervision_mask])
    print("check the label input for label input nodes: ", G.y_pad[G.label_input_mask])


    return G

def load_blogcatalog(data_name="blogcatalog", split_name="split_0.pt", train_percent=0.6, path="data/"):

    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + "blogcatalog/blogcatalog.mat")
    labels = mat['group']
    labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).long()

    # prepare the feature matrix
    features = torch.eye(labels.shape[0])

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
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
             edge_index=adj._indices(),
             y=labels)

    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask

    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/blogcatalog/blogcatalog_new.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    ################################# Split train to Supervision + Train ###################
    supervision_split_file_name = os.path.join(path, folder_name, split_name[:-3] + "_supervision_split.pt")
    print("supervision_split_file_name: ", supervision_split_file_name)
    # Check if supervision mask already exist
    if os.path.isfile(supervision_split_file_name):
        print("loading the supervision split...")
        # If the file exists, load the supervision mask from the file
        supervision_mask = torch.load(supervision_split_file_name)
        G.supervision_mask = supervision_mask
        # update the rest of the train node as input for label propagation
        G.label_input_mask = G.train_mask & ~G.supervision_mask

    # split the training nodes
    else:
        print("spliting the training nodes into supervision and label input...")
        # Get indices of training nodes
        train_indices = np.where(G.train_mask)[0]

        # Randomly split the training nodes into two groups
        np.random.shuffle(train_indices)
        split_index = len(train_indices) // 2

        # Create supervision_mask and update train_mask
        G.supervision_mask = np.zeros_like(G.train_mask, dtype=bool)
        G.supervision_mask[train_indices[:split_index]] = True
        # Convert the numpy arrays to PyTorch tensors
        G.supervision_mask = torch.from_numpy(G.supervision_mask)
        # Create label_input_mask as the difference between train_mask and supervision_mask
        G.label_input_mask = G.train_mask & ~G.supervision_mask

        torch.save(G.supervision_mask, supervision_split_file_name)


    # Check if the supervision_mask plus label_input_mask covers all the train_mask
    assert torch.all((G.supervision_mask | G.label_input_mask) == G.train_mask), "The supervision_mask plus label_input_mask do not cover all the train_mask."
    print("number of supervision nodes", torch.sum(G.supervision_mask).item())
    print("number of train nodes", torch.sum(G.train_mask).item())
    print("number of all nodes: ", G.x.shape[0])
    print("number of supervision nodes", torch.sum(G.label_input_mask).item())
    print("check unequal: ", int(torch.sum(G.train_mask).item() / 2), torch.sum(G.train_mask).item() / 2)


    # Check if the supervision_mask and label_input_mask are each of 50% size of the training nodes
    assert torch.sum(G.supervision_mask).item() == int(torch.sum(G.train_mask).item() / 2), "The supervision_mask is not 50% of the training nodes."

    # padd the input label matrix for supervision nodes
    G.y_pad[G.supervision_mask] = padding

    print("check the label input for supervision nodes: ", G.y_pad[G.supervision_mask])
    print("check the label input for label input nodes: ", G.y_pad[G.label_input_mask])

    return G


def load_yelp(data_name="yelp", split_name="split_0.pt", train_percent=0.6, path="data/"):

    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='data/yelp')
    data = dataset[0]
    labels = data.y
    features = data.x

    edge_index = data.edge_index

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
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
             y=labels,
             edge_index=edge_index)
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    
    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/yelp/yelp.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    ################################# Split train to Supervision + Train ###################
    supervision_split_file_name = os.path.join(path, folder_name, split_name[:-3] + "_supervision_split.pt")
    print("supervision_split_file_name: ", supervision_split_file_name)
    # Check if supervision mask already exist
    if os.path.isfile(supervision_split_file_name):
        print("loading the supervision split...")
        # If the file exists, load the supervision mask from the file
        supervision_mask = torch.load(supervision_split_file_name)
        G.supervision_mask = supervision_mask
        # update the rest of the train node as input for label propagation
        G.label_input_mask = G.train_mask & ~G.supervision_mask

    # split the training nodes
    else:
        print("spliting the training nodes into supervision and label input...")
        # Get indices of training nodes
        train_indices = np.where(G.train_mask)[0]

        # Randomly split the training nodes into two groups
        np.random.shuffle(train_indices)
        split_index = len(train_indices) // 2

        # Create supervision_mask and update train_mask
        G.supervision_mask = np.zeros_like(G.train_mask, dtype=bool)
        G.supervision_mask[train_indices[:split_index]] = True
        # Convert the numpy arrays to PyTorch tensors
        G.supervision_mask = torch.from_numpy(G.supervision_mask)
        # Create label_input_mask as the difference between train_mask and supervision_mask
        G.label_input_mask = G.train_mask & ~G.supervision_mask

        torch.save(G.supervision_mask, supervision_split_file_name)


    # Check if the supervision_mask plus label_input_mask covers all the train_mask
    assert torch.all((G.supervision_mask | G.label_input_mask) == G.train_mask), "The supervision_mask plus label_input_mask do not cover all the train_mask."
    print("number of supervision nodes", torch.sum(G.supervision_mask).item())
    print("number of train nodes", torch.sum(G.train_mask).item())
    print("number of all nodes: ", G.x.shape[0])
    print("number of supervision nodes", torch.sum(G.label_input_mask).item())
    print("check unequal: ", int(torch.sum(G.train_mask).item() / 2), torch.sum(G.train_mask).item() / 2)


    # Check if the supervision_mask and label_input_mask are each of 50% size of the training nodes
    assert torch.sum(G.supervision_mask).item() == int(torch.sum(G.train_mask).item() / 2), "The supervision_mask is not 50% of the training nodes."

    # padd the input label matrix for supervision nodes
    G.y_pad[G.supervision_mask] = padding

    print("check the label input for supervision nodes: ", G.y_pad[G.supervision_mask])
    print("check the label input for label input nodes: ", G.y_pad[G.label_input_mask])

    return G


def load_DBLP(data_name="dblp", split_name="split_0.pt", train_percent=0.6, path="data/dblp/"):

    print('Loading dataset ' + data_name + '...')

    features = torch.FloatTensor(np.genfromtxt(os.path.join(path, "features.txt"), delimiter=",", dtype=np.float64))
    labels = torch.FloatTensor(np.genfromtxt(os.path.join(path, "labels.txt"), delimiter=","))
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "dblp.edgelist"))).long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    folder_name = data_name + "_" + str(train_percent)
    file_path = os.path.join(path, folder_name, split_name)
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

    print("split loaded")

    G = Data(x=features,
             y=labels,
             edge_index=edge_index)
    
    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    
    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/dblp/dblp.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    ################################# Split train to Supervision + Train ###################
    supervision_split_file_name = os.path.join(path, folder_name, split_name[:-3] + "_supervision_split.pt")
    print("supervision_split_file_name: ", supervision_split_file_name)
    # Check if supervision mask already exist
    if os.path.isfile(supervision_split_file_name):
        print("loading the supervision split...")
        # If the file exists, load the supervision mask from the file
        supervision_mask = torch.load(supervision_split_file_name)
        G.supervision_mask = supervision_mask
        # update the rest of the train node as input for label propagation
        G.label_input_mask = G.train_mask & ~G.supervision_mask

    # split the training nodes
    else:
        print("spliting the training nodes into supervision and label input...")
        # Get indices of training nodes
        train_indices = np.where(G.train_mask)[0]

        # Randomly split the training nodes into two groups
        np.random.shuffle(train_indices)
        split_index = len(train_indices) // 2

        # Create supervision_mask and update train_mask
        G.supervision_mask = np.zeros_like(G.train_mask, dtype=bool)
        G.supervision_mask[train_indices[:split_index]] = True
        # Convert the numpy arrays to PyTorch tensors
        G.supervision_mask = torch.from_numpy(G.supervision_mask)
        # Create label_input_mask as the difference between train_mask and supervision_mask
        G.label_input_mask = G.train_mask & ~G.supervision_mask

        torch.save(G.supervision_mask, supervision_split_file_name)


    # Check if the supervision_mask plus label_input_mask covers all the train_mask
    assert torch.all((G.supervision_mask | G.label_input_mask) == G.train_mask), "The supervision_mask plus label_input_mask do not cover all the train_mask."
    print("number of supervision nodes", torch.sum(G.supervision_mask).item())
    print("number of train nodes", torch.sum(G.train_mask).item())
    print("number of all nodes: ", G.x.shape[0])
    print("number of supervision nodes", torch.sum(G.label_input_mask).item())
    print("check unequal: ", int(torch.sum(G.train_mask).item() / 2), torch.sum(G.train_mask).item() / 2)


    # Check if the supervision_mask and label_input_mask are each of 50% size of the training nodes
    assert torch.sum(G.supervision_mask).item() == int(torch.sum(G.train_mask).item() / 2), "The supervision_mask is not 50% of the training nodes."

    # padd the input label matrix for supervision nodes
    G.y_pad[G.supervision_mask] = padding

    print("check the label input for supervision nodes: ", G.y_pad[G.supervision_mask])
    print("check the label input for label input nodes: ", G.y_pad[G.label_input_mask])

    return G

def load_cora():
    # use validation for supervision, training nodes for label input

    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    label_one_hot = F.one_hot(data.y)

    G = Data(x=data.x,
             y=label_one_hot,
             edge_index=data.edge_index
            )
    G.uncode_label = data.y
    
    G.train_mask = data.train_mask
    G.val_mask = data.val_mask
    G.test_mask = data.test_mask
    G.n_id = torch.arange(G.x.shape[0])

    G.num_class = dataset.num_classes

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask

    # Pad the rows for test and validation nodes
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    # use validation nodes for supervision
    G.supervision_mask = G.val_mask

    # use training node true label as input
    G.label_input_mask = G.train_mask

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/cora/cora.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    return G

def load_cora_split_train():
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    label_one_hot = F.one_hot(data.y)

    G = Data(x=data.x,
             y=label_one_hot,
             edge_index=data.edge_index
            )
    G.uncode_label = data.y
    
    G.train_mask = data.train_mask
    G.val_mask = data.val_mask
    G.test_mask = data.test_mask
    G.n_id = torch.arange(G.x.shape[0])

    G.num_class = dataset.num_classes

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask

    # Pad the rows for test and validation nodes
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    # split the training nodes into input and supervision nodes
    train_indices = G.train_mask.nonzero(as_tuple=True)[0]
    train_labels = G.uncode_label[train_indices]
    class_indices = [train_indices[train_labels == i] for i in range(dataset.num_classes)]

    # Randomly select 10 nodes from each class for supervision
    selected_nodes = [indices[torch.randperm(len(indices))[:10]] for indices in class_indices]

    # Concatenate the selected nodes from each class
    selected_nodes = torch.cat(selected_nodes)

    # Create a mask for the supervision nodes
    supervision_mask = torch.zeros(G.num_nodes, dtype=bool)
    supervision_mask[selected_nodes] = True
    G.supervision_mask = supervision_mask

    # pad the labels fpr supervision nodes
    G.y_pad[G.supervision_mask] = padding
    print("check if the supervision nodes are padded:", G.y_pad[G.supervision_mask])


    # the other half of the training nodes
    label_input_mask = G.train_mask.clone().detach()
    label_input_mask[selected_nodes] = False
    G.label_input_mask = label_input_mask
    print("the labels of input nodes are not padded", G.y_pad[G.label_input_mask])
    print("chek the labels of other nodes are padded: ", G.y_pad[~G.label_input_mask])

    print("check of the label input nodes and the supervision nodes are chosen from training nodes",label_input_mask + supervision_mask == G.train_mask)

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/cora/cora.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    return G

def load_citeseer():
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]

    label_one_hot = F.one_hot(data.y)

    G = Data(x=data.x,
             y=label_one_hot,
             edge_index=data.edge_index
            )

    G.uncode_label = data.y
    
    G.train_mask = data.train_mask
    G.val_mask = data.val_mask
    G.test_mask = data.test_mask
    G.n_id = torch.arange(G.x.shape[0])

    G.num_class = dataset.num_classes
    G.num_nodes = G.x.shape[0]

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask

    # Pad the rows for test and validation nodes
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # pad the labels for val and test nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    
    G.supervision_mask = G.val_mask
    G.label_input_mask = G.train_mask
    print("the labels of input nodes are not padded", G.y_pad[G.label_input_mask])
    print("chek the labels of other nodes are padded: ", G.y_pad[G.supervision_mask])


    # Pad the rows for test and validation nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    ###################### load deep walk embeddings  ###################### 
    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)

    if len(isolated_nodes) > 0:
        # there are isolated nodes
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
        # has been padded, read from file
        if os.path.isfile('data/citeseer/citeseer_full.emb'):
            print("load random padding for those nodes from citeseer_full.emb.")
        # not padded, pad now
        else:
            emb_model = KeyedVectors.load_word2vec_format("data/citeseer/citeseer.emb", binary=False)
            embedding_dimensions = emb_model.vector_size
            
            for node in list(isolated_nodes):
                emb_model.add_vector(str(int(node)), np.random.rand(embedding_dimensions))

            emb_model.save_word2vec_format('data/citeseer/citeseer_full.emb')

        # load the full embeddings
        emb_model_full = KeyedVectors.load_word2vec_format("data/citeseer/citeseer_full.emb", binary=False)
        deep_walk_emb_full = np.asarray([emb_model_full[str(node)] for node in range(G.num_nodes)])
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb_full).float()


    # no isolated nodes, read the emb
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")
        print("loading the deepwalk emb...")
        emb_model = KeyedVectors.load_word2vec_format("data/citeseer/citeseer.emb", binary=False)
        deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()


    return G


def load_citeseer_split_train():
    dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')
    data = dataset[0]

    label_one_hot = F.one_hot(data.y)

    G = Data(x=data.x,
             y=label_one_hot,
             edge_index=data.edge_index
            )

    G.uncode_label = data.y
    
    G.train_mask = data.train_mask
    G.val_mask = data.val_mask
    G.test_mask = data.test_mask
    G.n_id = torch.arange(G.x.shape[0])

    G.num_class = dataset.num_classes
    G.num_nodes = G.x.shape[0]

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask

    # Pad the rows for test and validation nodes
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # pad the labels for val and test nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    ################ split the training nodes into input and supervision nodes ################
    train_indices = G.train_mask.nonzero(as_tuple=True)[0]
    train_labels = G.uncode_label[train_indices]
    class_indices = [train_indices[train_labels == i] for i in range(dataset.num_classes)]

    # Randomly select 10 nodes from each class for supervision
    selected_nodes = [indices[torch.randperm(len(indices))[:10]] for indices in class_indices]

    # Concatenate the selected nodes from each class
    selected_nodes = torch.cat(selected_nodes)

    # Create a mask for the supervision nodes
    supervision_mask = torch.zeros(G.num_nodes, dtype=bool)
    supervision_mask[selected_nodes] = True
    G.supervision_mask = supervision_mask

    # pad the labels of supervision nodes
    G.y_pad[G.supervision_mask] = padding
    print("check if the supervision nodes are padded:", G.y_pad[G.supervision_mask])


    # the label input nodes
    label_input_mask = G.train_mask.clone().detach()
    label_input_mask[selected_nodes] = False
    G.label_input_mask = label_input_mask
    print("the labels of input nodes are not padded", G.y_pad[G.label_input_mask])
    print("chek the labels of other nodes are padded: ", G.y_pad[G.supervision_mask])

    print("check of the label input nodes and the supervision nodes are chosen from training nodes",label_input_mask + supervision_mask == G.train_mask)

    # Pad the rows for test and validation nodes
    label_matrix = G.y.detach().float()
    label_matrix[val_test_mask] = padding
    G.y_pad = label_matrix

    ###################### load deep walk embeddings  ###################### 
    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)

    if len(isolated_nodes) > 0:
        # there are isolated nodes
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
        # has been padded, read from file
        if os.path.isfile('data/citeseer/citeseer_full.emb'):
            print("load random padding for those nodes from citeseer_full.emb.")
        # not padded, pad now
        else:
            emb_model = KeyedVectors.load_word2vec_format("data/citeseer/citeseer.emb", binary=False)
            embedding_dimensions = emb_model.vector_size
            
            for node in list(isolated_nodes):
                emb_model.add_vector(str(int(node)), np.random.rand(embedding_dimensions))

            emb_model.save_word2vec_format('data/citeseer/citeseer_full.emb')

        # load the full embeddings
        emb_model_full = KeyedVectors.load_word2vec_format("data/citeseer/citeseer_full.emb", binary=False)
        deep_walk_emb_full = np.asarray([emb_model_full[str(node)] for node in range(G.num_nodes)])
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb_full).float()


    # no isolated nodes, read the emb
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")
        print("loading the deepwalk emb...")
        emb_model = KeyedVectors.load_word2vec_format("data/citeseer/citeseer.emb", binary=False)
        deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
        G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()


    return G



######################################################################## Not changed from data_loader ###############################################################################

def load_humloc(data_name="HumanGo", path="data/"):

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

    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)


    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)
    if len(isolated_nodes) > 0:
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")


    # load deep walk embeddings, isolated nodes using random embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/HumanGo/humloc_full.emb", binary=False)
    # Get the dimensions of the embeddings
    embedding_dimensions = emb_model.vector_size
    # Use random embeddings for the isolated nodes
    for node in isolated_nodes:
        emb_model.add_vector(str(node), np.random.rand(embedding_dimensions))

    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()

    return G


def load_eukloc(data_name="EukaryoteGo", path="data/"):
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

    G.num_nodes = G.x.shape[0]
    G.n_id = torch.arange(G.num_nodes)

    # embedding the label for val and test
    val_test_mask = G.val_mask + G.test_mask
    padding = torch.full((1, G.y.shape[1]), 1 / G.y.shape[1]).float()

    # Pad the rows for test and validation nodes
    label_matrix = G.y.clone().detach()
    label_matrix[val_test_mask] = padding

    G.y_pad = label_matrix

    Label_train = G.y[G.train_mask].transpose(0, 1)
    # CN * NC
    LabelCor = torch.mm(Label_train, G.y[G.train_mask])
    G.LabelCor = row_normalize(LabelCor)

    # check isolated nodes
    isolated_nodes = check_isolated_nodes(G)
    if len(isolated_nodes) > 0:
        print(f"There are {len(isolated_nodes)} isolated nodes in the graph. Use random Padding for the isolated nodes.")
    else:
        print("There are no isolated nodes in the graph. No pre-process is performed")


    # load deep walk embeddings, isolated nodes using random embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/EukaryoteGo/eukloc_full.emb", binary=False)
    # Get the dimensions of the embeddings
    embedding_dimensions = emb_model.vector_size

    # Use random embeddings for the isolated nodes
    for node in isolated_nodes:
        emb_model.add_vector(str(node), np.random.rand(embedding_dimensions))

    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()


    return G





