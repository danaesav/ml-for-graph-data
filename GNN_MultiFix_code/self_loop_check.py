from data_loader import load_hyper_data

G = load_hyper_data(split_name="split_0.pt", train_percent=0.6, feature_noise_ratio=None, homo_level="homo06")


def contains_self_loops(edge_index):
    row, col = edge_index
    mask = row == col
    return mask.sum().item() > 0

has_self_loops = contains_self_loops(G.edge_index)
print(f'Contains self loops: {has_self_loops}')

# results: no self-loops in the edge_index