from data_loader import load_hyper_data



G = load_hyper_data(split_name=args.split_name, train_percent=args.train_percent, 
                    feature_noise_ratio=None, homo_level=0.6)


def contains_self_loops(edge_index):
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0

has_self_loops = contains_self_loops(edge_index)
print(f'Contains self loops: {has_self_loops}')