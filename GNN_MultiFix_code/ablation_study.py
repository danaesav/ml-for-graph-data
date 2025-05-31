from model import FPLPGCN_dw_linear, Local_Glbal_LC
from data_loader import *
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

if __name__ == "__main__":

    # Create and initialize the model
    #G = load_pcg(split_name="split_0.pt", train_percent=0.6)
    G = load_blogcatalog(split_name="split_1.pt", train_percent=0.6)
    input_dim = G.x.shape[1]
    hidden_dim = 256
    output_dim = G.y.shape[1]

    model = FPLPGCN_dw_linear(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1, dw_dim=G.deep_walk_emb.shape[1], multi_class=False)
    #model = Local_Glbal_LC(input_dim, hidden_dim, output_dim, num_gcn_layers=2, num_label_layers=1)

    # load the last checkpoint with the best model
    #model.load_state_dict(torch.load("checkpoints_blogcatalog_split1_linear/FPLPGCNblog00" + "___" + "split_0.pt" + '_checkpoint.pt'))
    model.load_state_dict(torch.load("checkpoints_blogcatalog_split1_linear/checkpoint_625.pt"))
    #output = model(x, y_pad, edge_index, G.deep_walk_emb)
    # print(output)
    # plt.imshow(output.detach().numpy()[G.test_mask][25:30, :], cmap='Blues')
    # plt.colorbar()
    # plt.show()
    # plt.savefig("")
    # are they the correct ones

    # weight of the linear layer
    weights = model.fusion_layer.weight.data   # (C,hidden_dim)
    # all 
    plt.imshow(weights.detach().numpy(), cmap="Blues", vmin=-0.6, vmax=0.6)
    plt.savefig("plot/linear_weights_all.png")

    # FP
    plt.imshow(weights[:, :256].detach().numpy(), cmap="Blues", vmin=-0.6, vmax=0.6)
    plt.savefig("plot/linear_weights_fp.png")


    # LP
    plt.imshow(weights[:, 256:295].detach().numpy(), cmap="Blues", vmin=-0.6, vmax=0.6)
    plt.savefig("plot/linear_weights_lp.png")

    # PE
    plt.imshow(weights[:, 295:].detach().numpy(), cmap="Blues", vmin=-0.6, vmax=0.6)
    plt.colorbar()
    plt.savefig("plot/linear_weights_pe.png")
    plt.cla()

    # check the predicted label






