from model import FPLPGCN_dw, FPLPGCN_dw_linear, FPLPGCN_dw_MLP
from data_loader import load_pcg, load_humloc, load_eukloc, load_yelp, load_blogcatalog, load_hyper_data, load_DBLP, load_cora, load_citeseer
from earlystopping import EarlyStopping
from args import get_args
from metrics import f1_loss, BCE_loss, rocauc_, ap_score
import torch.nn.functional as F
import numpy as np
import torch

if __name__ == "__main__":
    args = get_args()

    G_cora = load_cora()

    G_humloc = load_humloc()

    G_eukloc = load_eukloc()

    G_citeseer = load_citeseer()

    G_pcg = load_pcg(split_name=args.split_name, train_percent=args.train_percent)

    G_yelp = load_yelp(split_name=args.split_name, train_percent=args.train_percent)

    G_dblp = load_DBLP(split_name=args.split_name, train_percent=args.train_percent)

    G_blog = load_blogcatalog(split_name=args.split_name, train_percent=args.train_percent)

    G_hyper = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, feature_noise_ratio=args.feature_noise_ratio, homo_level=args.homo_level)


    print("################################################################################################")
    print("check G.y_pad:")
    print("pcg: ", G_pcg.y_pad)
    print("humloc: ", G_humloc.y_pad)
    print("eukloc: ", G_eukloc.y_pad)
    print("yelp: ", G_yelp.y_pad)
    print("blog:" , G_blog.y_pad)
    print("hyper: ", G_hyper.y_pad)
    print("dblp: ", G_dblp.y_pad)
    print("cora: ", G_cora.y_pad)
    print("cora supervision nodes labels", G_cora.y_pad[G.supervision_mask])
    print("cora input nodes labels", G_cora.y_pad[G.train_mask-G.supervision_mask])

    print("citeseer supervision nodes labels", G_citeser.y_pad[G.supervision_mask])
    print("citeseer input nodes labels", G_citeseer.y_pad[G.train_mask-G.supervision_mask])

    print("################################################################################################")


