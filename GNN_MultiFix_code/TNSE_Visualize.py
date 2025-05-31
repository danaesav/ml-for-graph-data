from model import FPLPGCN_dw_MLP, FPLPGCN_dw_linear
from data_loader import *
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Create and initialize the model
    #G = load_pcg(split_name="split_0.pt", train_percent=0.6)
    #G = load_blogcatalog(split_name="split_0.pt", train_percent=0.6)
    G = load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=0.2)

    input_dim = G.x.shape[1]
    print(input_dim)
    hidden_dim = 128
    output_dim = G.y.shape[1]

    #model = FPLPGCN(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1)
    model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=5, dw_dim=G.deep_walk_emb.shape[1])
    print(model)

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load("FPLPGCN_linear_hyper_0.2_S0___split_0.pt_checkpoint.pt"))

    output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    print("output dim", output.shape)

    # the embeddings of FP and LP
    FP_LP_emb = torch.cat((model.FP, model.LP), dim=1)
    print("FPLP dim", FP_LP_emb.shape)

    # the embeddings from dw
    dw_emb = model.dw_emb
    print("dw dim", dw_emb.shape)

    # when they are combined together
    complete_emb = model.all_emb

    # Create a PCA object
    tsne1 = TSNE(n_components=2)
    tsne2 = TSNE(n_components=2)
    tsne3 = TSNE(n_components=2)

    # Fit the PCA model and transform the embeddings to 2D
    fplp_2d = tsne1.fit_transform(FP_LP_emb.detach().numpy())
    dw_2d = tsne2.fit_transform(dw_emb.detach().numpy())
    cmplt_2d = tsne3.fit_transform(complete_emb.detach().numpy())

    # Plot the 2D embeddings
    plt.scatter(fplp_2d[:, 0], fplp_2d[:, 1])
    # Save the plot to a file
    plt.savefig("fplp.png")  

    # Plot the 2D embeddings
    plt.scatter(dw_2d[:, 0], dw_2d[:, 1])
    # Save the plot to a file
    plt.savefig("dw.png")  

    # Plot the 2D embeddings
    plt.scatter(cmplt_2d[:, 0], cmplt_2d[:, 1])
    # Save the plot to a file
    plt.savefig("all.png")  

    print("TNSE visualization saved")






