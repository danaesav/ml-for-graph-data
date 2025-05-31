import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def get_args():
    parser = argparse.ArgumentParser(description="Your Script Description")

    # Add command-line arguments
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--data_path', type=str, default='data/', help='Path to the data directory')
    parser.add_argument('--data_name', type=str, default='hyper', help='Name of the dataset')
    parser.add_argument('--split_name', type=str, default='split_0.pt', help='Random split of the data')

    parser.add_argument('--model_name', type=str, default='FPLPGCN_linear_', help='Name of the model')

    parser.add_argument('--fp', type=str2bool, nargs='?', help='enable feature propagation')
    parser.add_argument('--lp', type=str2bool, nargs='?', help='enable label propagation')
    parser.add_argument('--pe', type=str2bool, nargs='?',help='enable positional encoding')

    parser.add_argument('--num_fp', type=int, default=2, help='Number of feature propagation')
    parser.add_argument('--num_lp', type=int, default=5, help='Number of label propagation')

    parser.add_argument('--train_percent', type=float, default=0.6, help='percentage of data used for training')

    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden unit')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--patience', type=int, default=100, help='Patience for early stopping')
    parser.add_argument('--feature_noise_ratio', type=float, default=None,
                        help='The ratios of the relevant features and irrelevant features')
    parser.add_argument('--homo_level', type=str, default=None,
                        help='The homophily level of the synthetic dataset')
    
    parser.add_argument('--feat_type',type=str, default='all', help='Type of features to be used')


    args = parser.parse_args()
    return args
