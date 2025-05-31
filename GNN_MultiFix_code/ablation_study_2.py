from model import FPLPGCN, Local_Glbal_LC, FPLPGCN_dw, FPLPGCN_dw_linear
from data_loader import *
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == "__main__":

    # Create and initialize the model
    #G = load_pcg(split_name="split_0.pt", train_percent=0.6)
    #G = load_blogcatalog(split_name="split_0.pt", train_percent=0.6)
    G = load_hyper_data(split_name="split_0.pt", train_percent=0.6)

    input_dim = G.x.shape[1]
    hidden_dim = 256
    output_dim = G.y.shape[1]

    #model = FPLPGCN(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1)
    model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=10)

    # load the last checkpoint with the best model
    #model.load_state_dict(torch.load("FPLPGCN" + "_hyper_fn00___" + "split_0.pt" + '_checkpoint.pt'))
    #linear model
    model.load_state_dict(torch.load("linear_FPLPGCN" + "___" + "split_0.pt" + '_checkpoint.pt'))

    output = model(G.x, G.y_pad, G.edge_index, G.deep_walk_emb)
    # print(output)
    # plt.imshow(output.detach().numpy()[G.test_mask][25:30, :], cmap='Blues')
    # plt.colorbar()
    # plt.show()
    # are they the correct ones

    #weight of the linear layer
    weights = model.fusion_layer.weight.data   # (C,hidden_dim)
    plt.imshow(weights.detach().numpy()[:, :], cmap="Blues") # 0-256, 256-276, 276-340
    
    plt.colorbar()
    plt.show()

    # check the predicted label






