from baseline_models import GCN, GAT, SAGE_sup, H2GCN, GCN_LPA, MLP, LANC, GatedGCNNet, FSGNN
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import torch.nn.functional as F
import numpy as np
import torch
from utils import glorot
from torch_geometric.utils import add_self_loops
from torch_geometric.loader import NeighborLoader
import copy
from torch.utils.data import DataLoader
from torch_geometric.utils import degree
# for LSPE
from utils import init_positional_encoding
import dgl
from utils import compute_identity, compute_identity_sparse
######################### standard train function #########################
def model_train():

    # train model
    model.train()
    optimizer.zero_grad()

    # forward pass
    if args.model_name.startswith("MLP"):
        output = model(x)
    else:
        output = model(x, edge_index)

    loss_train = BCE_loss(output[train_mask], labels[train_mask])

    # evaluation
    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = rocauc_(labels[train_mask], output[train_mask])

    # back propagation
    loss_train.backward()
    optimizer.step()

    return loss_train, micro_train, macro_train, roc_auc_train_macro
###########################################################################


################################################## standard test function ##################################################
@torch.no_grad()
def model_test():

    # test model
    model.eval()

    # forward pass
    if args.model_name.startswith("MLP"):
        output = model(x)
    else:
        output = model(x, edge_index)


    # evaluation
    loss_val = BCE_loss(output[val_mask], labels[val_mask])

    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = rocauc_(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = rocauc_(labels[test_mask], output[test_mask])

    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test
#############################################################################################################################


################################################## batch train ##################################################
def batch_train(loader):
    total_loss = 0
    model.train()

    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        labels_target = batch.y[:batch.batch_size]
        loss = BCE_loss(out, labels_target)

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * out.size(0)

    return total_loss / torch.sum(G.train_mask)
#############################################################################################################################

################################################## batch test ##################################################
@torch.no_grad()
def batch_test(subgraph_loader):

    output = model.inference(G.x, subgraph_loader)

    loss_val = BCE_loss(output[val_mask], labels[val_mask])
    micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
    roc_auc_val_macro = rocauc_(labels[val_mask], output[val_mask])

    micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
    roc_auc_test_macro = rocauc_(labels[test_mask], output[test_mask])
    ap_test = ap_score(labels[test_mask], output[test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test
#############################################################################################################################


######################### gcnlpa train #########################
def train_gcnlpa():
    model.train()
    optimizer.zero_grad()
    output, predicted_labels = model(G.x, G.y_pad, G.edge_index)

    #gcn_loss
    gcn_loss = BCE_loss(output[G.train_mask], G.y[G.train_mask])
    #lpa_loss
    lpa_loss = BCE_loss(predicted_labels[G.train_mask], G.y[G.train_mask])

    loss = gcn_loss + lpa_loss
    loss.backward()

    optimizer.step()
    return float(loss)
###########################################################################


########################################################################### gcnlpa test ###########################################################################
@torch.no_grad()
def test_gcnlpa():
    model.eval()
    output, predicted_labels = model(G.x, G.y_pad, G.edge_index)

    micro_train, macro_train = f1_loss(G.y[G.train_mask], output[G.train_mask])
    roc_auc_train_macro = rocauc_(G.y[G.train_mask], output[G.train_mask])
    ap_train = ap_score(G.y[G.train_mask], output[G.train_mask])

    gcn_loss_val = BCE_loss(output[G.val_mask], G.y[G.val_mask])
    lpa_loss_val = BCE_loss(predicted_labels[G.val_mask], G.y[G.val_mask])
    loss_val = gcn_loss_val + lpa_loss_val

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output[G.val_mask])
    roc_auc_val_macro = rocauc_(G.y[G.val_mask], output[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output[G.test_mask])
    roc_auc_test_macro = rocauc_(G.y[G.test_mask], output[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output[G.test_mask])

    return micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test
###################################################################################################################################################################


########################################################################### LANC train ###########################################################################
def train_lanc(train_loader):
    un_lbl = torch.arange(0, G.y.shape[1])

    outs1 = []
    total_loss = 0
    for idx in train_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, out2 = model.forward(x, y)
        outs1.append(out1)

        loss_train = BCE_loss(out1, G.y[idx]) + F.cross_entropy(out2, un_lbl)
        loss_train.backward()
        optimizer.step()

        total_loss += float(loss_train) * len(idx)

    output1 = torch.cat(outs1, dim=0)

    micro_train, macro_train = f1_loss(G.y[G.train_mask], output1)
    roc_auc_train_macro = rocauc_(G.y[G.train_mask], output1)

    ap_train = ap_score(G.y[G.train_mask], output1)

    return total_loss/G.num_nodes, micro_train, macro_train, roc_auc_train_macro, ap_train
####################################################################################################################################################################

########################################################################### LANC test ###########################################################################
def test_lanc(eva_loader):
    un_lbl = torch.arange(0, G.y.shape[1])
    # all embedding output
    outs1 = []
    for idx in eva_loader:
        x = G.x[idx]
        y = G.lbl_emb
        out1, output2 = model.forward(x, y)
        outs1.append(out1)

    # embedding prediction
    output1 = torch.cat(outs1, dim=0)
    # calculate loss
    loss_val = BCE_loss(output1[G.val_mask], G.y[G.val_mask]) + F.cross_entropy(output2, un_lbl)

    micro_val, macro_val = f1_loss(G.y[G.val_mask], output1[G.val_mask])
    roc_auc_val_macro = rocauc_(G.y[G.val_mask], output1[G.val_mask])
    ap_val = ap_score(G.y[G.val_mask], output1[G.val_mask])

    micro_test, macro_test = f1_loss(G.y[G.test_mask], output1[G.test_mask])
    roc_auc_test_macro = rocauc_(G.y[G.test_mask], output1[G.test_mask])
    ap_test = ap_score(G.y[G.test_mask], output1[G.test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test
####################################################################################################################################################################


########################################################################### LSPE train ###########################################################################
def train_lspe(data_loader):

    device = args.device
    model.train()
    
    epoch_loss = 0
    nb_data = 0
    
    y_true = []
    y_pred = []

    for iter, batch in enumerate(data_loader):
        optimizer.zero_grad()

        # batch_graphs = batch_graphs.to(device)
        batch_graph = dgl.graph((batch.edge_index[0], batch.edge_index[1]))
        batch_x = batch.x.to(device)
        # no edge features
        num_edges = batch.edge_index.shape[1]
        batch_e = torch.zeros(num_edges, batch.x.shape[1])
        batch_labels = batch.y.to(device)
        # batch_snorm_n = batch_snorm_n.to(device)
        batch_snorm_n = None

        ############################# PE #################################
        try:
            batch_pos_enc = batch.pe.to(device)
        except KeyError:
            batch_pos_enc = None
            print("PE Initial Not Successful!!!")
        ##################################################################

        if model.pe_init == 'lap_pe':
            sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
            
        batch_pred, __ = model.forward(batch_graph, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        batch_pred = batch_pred[:batch.batch_size]
        del __
        
        ################################ Graph Classification ##################################
        # ignore nan labels (unlabeled) when computing training loss
        # is_labeled = batch_labels == batch_labels
        # loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        #######################################################################################################

        ################################# Node Classification ################################
        # only calculate the loss on training nodes
        loss = model.loss(batch_pred, batch_labels[:batch.batch_size])
        #######################################################################################################
        
        loss.backward()
        optimizer.step()
        
        y_true.append(batch_labels[:batch.batch_size].view(batch_pred.shape))
        y_pred.append(batch_pred)
        
        epoch_loss += loss.detach().item() * batch_pred.size(0)
        nb_data += batch_pred.size(0)
    
    epoch_loss /= torch.sum(G.train_mask)
    
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    # compute performance metric: AP
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    # evaluation
    micro_train, macro_train = f1_loss(y_true[G.train_mask], y_pred[G.train_mask])
    roc_auc_train_macro = rocauc_(y_true[G.train_mask], y_pred[G.train_mask])
    ap_train = ap_score(y_true[G.train_mask], y_pred[G.train_mask])
    return epoch_loss, micro_train, macro_train, roc_auc_train_macro, ap_train

####################################################################################################################################################################

########################################################################### LSPE test ###########################################################################
def test_lspe(data_loader):

    device = args.device
    model.eval()

    epoch_loss = 0
    nb_data = 0

    y_true = []
    y_pred = []
    
    out_graphs_for_lapeig_viz = []
    
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            batch_graph = dgl.graph((batch.edge_index[0], batch.edge_index[1]))
            batch_x = batch.x.to(device)
            num_edges = batch.edge_index.shape[1]
            batch_e = torch.zeros(num_edges, batch.x.shape[1])
            batch_labels = batch.y.to(device)
            #batch_snorm_n = batch_snorm_n.to(device)
            batch_snorm_n = None
            
            try:
                batch_pos_enc =  batch.pe.to(device)
            except KeyError:
                batch_pos_enc = None
                print("PE Initial Not Successful!!!")
            
            batch_pred, batch_g = model.forward(batch_graph, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
            batch_pred = batch_pred[:batch.batch_size]
            
            # ignore nan labels (unlabeled) when computing loss
            # is_labeled = batch_labels == batch_labels
            # loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
            
            y_true.append(batch_labels[:batch.batch_size].view(batch_pred.shape))
            y_pred.append(batch_pred)

            # epoch_loss += loss.detach().item()
            nb_data += batch_labels[:batch.batch_size].size(0)
            
            if batch_g is not None:
                out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
            else:
                out_graphs_for_lapeig_viz = None
            
        # epoch_loss /= (iter + 1)

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    # compute performance metric AP
    loss_val = BCE_loss(y_pred[G.val_mask], y_true[G.val_mask])

    micro_val, macro_val = f1_loss(y_true[G.val_mask], y_pred[G.val_mask])
    roc_auc_val_macro = rocauc_(y_true[G.val_mask], y_pred[G.val_mask])
    ap_val = ap_score(y_true[G.val_mask], y_pred[G.val_mask])

    micro_test, macro_test = f1_loss(y_true[G.test_mask], y_pred[G.test_mask])
    roc_auc_test_macro = rocauc_(y_true[G.test_mask], y_pred[G.test_mask])
    ap_test = ap_score(y_true[G.test_mask], y_pred[G.test_mask])

    return loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test
        
    # return epoch_loss, return_perf, out_graphs_for_lapeig_viz
####################################################################################################################################################################

########################################################################### FSGNN ###########################################################################

############################## Train ##################################
def train_step(model, optimizer, labels, list_mat):
    model.train()
    optimizer.zero_grad()
    output = model(list_mat, layer_norm)
    loss_train = BCE_loss(output[train_mask], labels[train_mask].to(device))
    loss_train.backward()
    optimizer.step()

    # evaluation
    micro_train, macro_train = f1_loss(labels[train_mask], output[train_mask])
    roc_auc_train_macro = rocauc_(labels[train_mask], output[train_mask])

    return loss_train.item(), micro_train, macro_train, roc_auc_train_macro

########################################################################

############################## Test ##################################

def test_step(model,labels,list_mat):

    model.eval()
    with torch.no_grad():
        output = model(list_mat, layer_norm)
        # evaluation
        loss_val = BCE_loss(output[val_mask], labels[val_mask])

        micro_val, macro_val = f1_loss(labels[val_mask], output[val_mask])
        roc_auc_val_macro = rocauc_(labels[val_mask], output[val_mask])

        micro_test, macro_test = f1_loss(labels[test_mask], output[test_mask])
        roc_auc_test_macro = rocauc_(labels[test_mask], output[test_mask])

        ap_test = ap_score(labels[test_mask], output[test_mask])

        return loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, ap_test
########################################################################

####################################################################################################################################################################


if __name__ == "__main__":

    #################################### Hyperparameter Setting #####################################
    args = get_args()
    print("###################the parameter setting################")
    args.multi_class = False
    print(args)
    
    ################################### import data #############################################
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
        G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                            feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)

    elif args.data_name == "dblp":
        G = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    elif args.data_name == "cora":
        G = load_cora()

    elif args.data_name == "citeseer":
        G = load_citeseer()


    ############################################ Check Hyperparameters #############################################
    input_dim = G.x.shape[1]
    hidden_dim = args.hidden_dim

    # multi-class
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        output_dim = G.num_class
        print("this is a multi-class dataset, output dim is number of classes")

    # multi-label
    else:
        output_dim = G.y.shape[1]

    x = G.x
    edge_index = G.edge_index
    y_pad = G.y_pad

    print("check if the val and test are padded: ", y_pad[G.val_mask], y_pad[G.test_mask])
    print("check if the true labels are unpadded: ", G.y[G.val_mask], G.y[G.test_mask])

    # multi-class multl-label indicater
    if args.data_name == "cora" or args.data_name == "citeseer" or args.data_name == "pubmed":
        args.multi_class = True
        print("args.multi_class set to True")
    else:
        args.multi_class = False


    ############################################## Model Initialization ################################################
    if args.model_name.startswith("GCNLPA"):

        # preprocessing required by GCN-LPA
        G.edge_index = add_self_loops(G.edge_index)[0]
        edge_weights = glorot(shape=G.edge_index.shape[1])
        G.edge_weights = edge_weights

        model = GCN_LPA(in_channels=G.x.shape[1], hidden_channels=hidden_dim, out_channels=G.y.shape[1], num_gcn=2, lpa_iter=5, edge_weights=G.edge_weights, multi_class=args.multi_class)
       
    
    elif args.model_name.startswith("SAGE"):
        model = SAGE_sup(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=args.multi_class)
        train_loader = NeighborLoader(G, input_nodes=G.train_mask, num_neighbors=[10, 5], batch_size=1024, shuffle=False)
        subgraph_loader = NeighborLoader(copy.copy(G), input_nodes=None, num_neighbors=[-1], shuffle=False, batch_size=1024)

    elif args.model_name.startswith("GAT"):
        model = GAT(in_channels=G.x.shape[1], class_channels=G.y.shape[1], multi_class=args.multi_class)

    elif args.model_name.startswith("MLP"):
        model = MLP(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=args.multi_class)

    elif args.model_name.startswith("H2GCN"):
        model = H2GCN(nfeat=G.x.shape[1], nhid=hidden_dim, nclass=G.y.shape[1], multi_class=args.multi_class)

    elif args.model_name.startswith("GCN"):
        model = GCN(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=args.multi_class)

    elif args.model_name.startswith("LANC"):
        # preprocess for LANC
        features = x
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]),
                                  (x.shape[0], x.shape[0]))
        adj_dense = adj.to_dense()
        G.adj = adj

        lbl_emb = torch.arange(G.y.shape[1]).long()
        G.lbl_emb = lbl_emb
        num_nodes = G.y.shape[0]
        G.n_id = torch.arange(num_nodes)
        print(G.n_id)
        # maximum degree
        degree = 0

        for row in adj_dense:
            deg = torch.flatten(torch.nonzero(row)).shape[0]
            if deg > degree:
                degree = deg

        # prepare the feature matrix
        attrs = []
        for i in range(num_nodes):
            neigh_ind = torch.flatten(torch.nonzero(adj_dense[i]))
            if len(neigh_ind) < degree:
                num_to_pad = degree - len(neigh_ind)
                padding = torch.zeros(num_to_pad, features.shape[1])
                # featues of neighbors
                attr = features[neigh_ind]
                attr = torch.vstack((attr, padding))
            else:
                attr = features[neigh_ind]
                # featues of neighbors
            attrs.append(attr)

        G.x = torch.stack(attrs)

        print(G.x.shape)
        model = LANC(in_channels=G.x.shape[2], class_channels=G.y.shape[1], num_label=G.lbl_emb.shape[0], multi_class=False)

        train_loader = DataLoader(G.n_id[G.train_mask], shuffle=False, batch_size=64, num_workers=0)
        eva_loader = DataLoader(G.n_id, shuffle=False, batch_size=64, num_workers=0)

    elif args.model_name.startswith("LSPE"):
        # match the original implementation, prepare args_name
        args.input_dim = G.x.shape[1]
        args.out_dim = G.y.shape[1]
        args.n_classes = G.y.shape[1]
        args.dropout = 0.0
        args.L = 2
        args.readout = "mean"
        args.batch_norm = True
        args.residual = True
        args.device = "cpu"
        args.pe_init = "rand_walk"
        args.use_lapeig_loss = False
        args.lambda_loss = True
        args.alpha_loss = True
        args.pos_enc_dim = 64

        ######################## Initialization ##########################
        # convert G into dgl graph
        G.pe = init_positional_encoding(G, pos_enc_dim=64)
        print("ini PE: ", G.pe.shape)
        model = GatedGCNNet(args)

        train_loader = NeighborLoader(G, shuffle=False, num_neighbors=[-1, -1], batch_size=64, num_workers=0)
        eva_loader = NeighborLoader(G, shuffle=False, num_neighbors=[-1, -1], batch_size=64, num_workers=0)

    elif args.model_name.startswith("IDGNN"):
        # preprocessing fro IDGNN, calculate the indenty info
        if args.data_name == "yelp":
            # not possible to use adj_dense
            identity = compute_identity_sparse(G.edge_index, n=G.x.shape[0], k=3)
        else:
            # use adj_dense
            identity = compute_identity(G.edge_index, n=G.x.shape[0], k=3) # (n, k)
        print("identity dim: ", identity.shape)
        # inject the identity info into feat
        G.x = torch.cat((G.x, identity), dim=1)

        # use normal GCN
        model = GCN(in_channels=G.x.shape[1], hidden_channels=hidden_dim, class_channels=G.y.shape[1], multi_class=args.multi_class)

    elif args.model_name.startswith("FSGNN"):
        device = "cpu"
        args.layer_norm = 1
        layer_norm = bool(int(args.layer_norm))
        feat_type = args.feat_type
        args.num_layer = 3

        args.w_fc1 = 0.0005
        args.w_fc2 = 0.0005
        args.w_att = 0.0005

        args.lr_fc = 0.02
        args.lr_att = 0.02

        # check the parameter setting
        print("################### Final Parameter Check ################")
        args.multi_class = False
        print(args)
        # added hyper-parameter

        # check if G.adj
        if hasattr(G, 'adj'):
            print("adj is saved in G")
        else:
            print("G has only edge_index, constructing adj")
            num_nodes = len(G.n_id)
            # Create a sparse adjacency matrix
            G.adj = torch.sparse_coo_tensor(indices=G.edge_index, values=torch.ones(G.edge_index.size(1)), size=(num_nodes, num_nodes))

        # Add self-loops to the edge_index
        edge_index_loop, _ = add_self_loops(G.edge_index, num_nodes=G.num_nodes)
        # Create a sparse adjacency matrix with self-loops
        G.adj_i = torch.sparse_coo_tensor(indices=edge_index_loop, values=torch.ones(edge_index_loop.size(1)), size=(G.num_nodes, G.num_nodes))
        print(G.adj_i)

        # preprocessing
        list_mat=[]
        list_mat.append(G.x)
        no_loop_mat = G.x
        loop_mat = G.x

        for ii in range(args.num_layer):
            no_loop_mat = torch.spmm(G.adj, no_loop_mat)
            loop_mat = torch.spmm(G.adj_i, loop_mat)
            list_mat.append(no_loop_mat)
            list_mat.append(loop_mat)

        # Select X and self-looped features 
        if feat_type == "homophily":
            select_idx = [0] + [2*ll for ll in range(1, args.num_layer+1)]
            list_mat = [list_mat[ll] for ll in select_idx]

        #Select X and no-loop features
        elif feat_type == "heterophily":
            select_idx = [0] + [2*ll-1 for ll in range(1, args.num_layer+1)]
            list_mat = [list_mat[ll] for ll in select_idx]
        

        model = FSGNN(nfeat=G.x.shape[1], nlayers=len(list_mat), nhidden=args.hidden_dim, nclass=G.y.shape[1], dropout=0.5)

    else:
        raise OSError("model not defined, add new baseline model in baseline_models.py")

    print(model)
    #######################################################################################################################################

    #################################################################################### Train ############################################################################

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    early_stopping = EarlyStopping(model_name=args.model_name, split_name=args.split_name, patience=args.patience, verbose=True)

    # set optimizer for FSGNN
    if args.model_name.startswith("FSGNN"):
        
        optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
        {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
        ]

        optimizer = torch.optim.Adam(optimizer_sett)

    x = G.x
    labels = G.y
    edge_index = G.edge_index
    train_mask = G.train_mask
    val_mask = G.val_mask
    test_mask = G.test_mask

    if args.model_name.startswith("SAGE"):
        for epoch in range(1, args.epochs):
            loss_train = batch_train(train_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        _, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = batch_test(subgraph_loader)
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
              )

    elif args.model_name.startswith("GCNLPA"):
        for epoch in range(1, args.epochs):
            loss_train = train_gcnlpa()
            micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_gcnlpa()
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f},'
                f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f}, '
                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
                # f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                # f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        micro_train, macro_train, roc_auc_train_macro, ap_train, loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_gcnlpa()
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )

    elif args.model_name.startswith("LANC"):
        for epoch in range(1, args.epochs):

            loss_train, micro_train, macro_train, roc_auc_train_macro, ap_train = train_lanc(train_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lanc(eva_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f}, '
                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
                f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lanc(eva_loader)

        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
              f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {ap_test:.4f}, '
             )

    elif args.model_name.startswith("LSPE"):

        for epoch in range(1, args.epochs):
            loss_train, micro_train, macro_train, roc_auc_train_macro, ap_train = train_lspe(train_loader)
            loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lspe(eva_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f}, '
                f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f}, '
                f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                f'Test Average Precision Score: {ap_test:.4f}, '
                )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        loss_val, micro_val, macro_val, roc_auc_val_macro, ap_val, micro_test, macro_test, roc_auc_test_macro, ap_test = test_lspe(eva_loader)

        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
              f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {ap_test:.4f}, '
             )
        

    elif args.model_name.startswith("FSGNN"):

        for epoch in range(1, args.epochs):
            loss_train, micro_train, macro_train, roc_auc_train_macro = train_step(model, optimizer, labels, list_mat)
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = test_step(model, labels,list_mat)
            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        print("Optimization Finished!")

        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        #loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
        loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = test_step(model, labels,list_mat)
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
             )


    # other models
    else:
        for epoch in range(1, args.epochs):
            loss_train, micro_train, macro_train, roc_auc_train_macro = model_train()
            loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()

            print(f'Epoch: {epoch:03d}, Loss: {loss_train:.10f}, '
                  f'Train micro: {micro_train:.4f}, Train macro: {macro_train:.4f} '
                  f'Val micro: {micro_val:.4f}, Val macro: {macro_val:.4f} '
                  f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
                  #f'Train ROC-AUC micro: {roc_auc_train_micro:.4f}, '
                  f'Train ROC-AUC macro: {roc_auc_train_macro:.4f} '
                  #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
                  f'Val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
                  #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
                  f'Test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
                  f'Test Average Precision Score: {test_ap:.4f}, '
                  )
            early_stopping(loss_val, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("Optimization Finished!")

        #######################################################################################################################################

        ############################################################ Final Evaluation ##########################################################
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load(args.model_name + "___" + args.split_name + '_checkpoint.pt'))
        #loss_val, micro_val, macro_val, roc_auc_val_micro, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_micro, roc_auc_test_macro, test_ap = model_test()
        loss_val, micro_val, macro_val, roc_auc_val_macro, micro_test, macro_test, roc_auc_test_macro, test_ap = model_test()
        print(f'Test micro: {micro_test:.4f}, Test macro: {macro_test:.4f} '
              #f'Val ROC-AUC micro: {roc_auc_val_micro:.4f}, '
              f'val ROC-AUC macro: {roc_auc_val_macro:.4f}, '
              #f'Test ROC-AUC micro: {roc_auc_test_micro:.4f}, '
              f'test ROC-AUC macro: {roc_auc_test_macro:.4f}, '
              f'Test Average Precision Score: {test_ap:.4f}, '
             )
        #######################################################################################################################################






