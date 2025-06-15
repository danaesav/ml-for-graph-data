import numpy as np
import torch as th
import os

from tqdm import tqdm
from torch_geometric_temporal import temporal_signal_split
from matplotlib import pyplot as plt
from datetime import datetime

from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig
from model_results import ModelResults
from models.MultiLabelEvolveGCN import MultiLabelEvolveGCN
from models.TemporalMultiFix import TemporalMultiFix
from dataset_loader import DatasetLoader
from utils import metrics

DEVICE = th.device('cuda' if th.cuda.is_available() else 'cpu')


PARAM = {'NUM_NODES' : 3000,  # Must match N in generator config
        'NUM_REL_FEATURES' : 10,
        'NUM_IRR_FEATURES' : 5,
        'NUM_RED_FEATURES' : 0,
        'NUM_LABELS' : 20,  # q = number of hyperspheres
        'NUM_TIMESTEPS' : 30,  # horizon
        'EPOCHS' : 500,
        'LR_MLEGCN': 2e-2,
        'LR_TMF': 8e-3,
        'THRESHOLD' : 0.5,  # for classification
        'REPEATS' : 5,
        'ALPHA' : 5,
        'EMBEDDING_DIM' : 32,
        'TRAIN_RATIO' : 0.6,
        'VALIDATION_RATIO' : 0.2,
        'TEST_RATIO' : 0.2,
        'FILENAME' : '.\\data\\base_graphs',
        'BASEFILE' : '.\\data\\base_graphs_2025-06-12_02-49-34', #None for new base graph
        'EXPERIMENT_PATH': '.\\data',
        'DATA_FILE': 'results',
        'IMAGE_FILE': 'image',
    }


def temporal_signal_split_list(data, train_ratio):
    train_snapshots = int(train_ratio * len(data))

    train_iterator = data[0:train_snapshots]
    test_iterator = data[train_snapshots:]

    return train_iterator, test_iterator


def load_data(dataset_loader, param):
    dataset, embeddings, inter_homophily, intra_homophily = dataset_loader.get_dataset()

    training_loop_ratio = param["TRAIN_RATIO"]+param["VALIDATION_RATIO"]

    tmp_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=training_loop_ratio)
    tmp_embedding, test_embedding = temporal_signal_split_list(embeddings, train_ratio=training_loop_ratio)

    train_dataset, validation_dataset = temporal_signal_split(tmp_dataset, train_ratio=param["TRAIN_RATIO"]/training_loop_ratio)
    train_embedding, validation_embedding = temporal_signal_split_list(tmp_embedding, train_ratio=param["TRAIN_RATIO"]/training_loop_ratio)


    datasets = {'train_data' : train_dataset,
                'validation_data' : validation_dataset,
                'test_data' : test_dataset,
                'train_emb' : train_embedding,
                'validation_emb' : validation_embedding,
                'test_emb' : test_embedding,
                'inter_homophily' : inter_homophily,
                'intra_homophily' : intra_homophily,
                }

    return datasets
    

def initialize_models(param):
    # deepwalk_embeddings = None #shape(TIME, NUM_NODES, NUM_FEATURES)

    # Temporal MultiFix with Deepwalk embeddings
    model_tmf_dw = TemporalMultiFix(
        input_dim=param["NUM_REL_FEATURES"] + param["NUM_IRR_FEATURES"] + param["NUM_RED_FEATURES"],
        num_of_nodes=param["NUM_NODES"],
        output_dim=param["NUM_LABELS"],
        dw_dim=param["EMBEDDING_DIM"],).to(DEVICE)

    optimizer_tmf_dw = th.optim.Adam(model_tmf_dw.parameters(), lr=param["LR_TMF"])

    tmf_dw = {'model':model_tmf_dw,
              'optimizer':optimizer_tmf_dw,
              'type': "tmf_dw"}


    # Temporal MultiFix without Deepwalk Embeddings
    model_tmf = TemporalMultiFix(
        input_dim=param["NUM_REL_FEATURES"] + param["NUM_IRR_FEATURES"] + param["NUM_RED_FEATURES"],
        num_of_nodes=param["NUM_NODES"],
        output_dim=param["NUM_LABELS"]).to(DEVICE)

    optimizer_tmf = th.optim.Adam(model_tmf.parameters(), lr=param["LR_TMF"])

    tmf = {'model': model_tmf,
          'optimizer': optimizer_tmf,
            'type': "tmf"}


    # MultiLabel EvolveGCN
    model_mlegcn = MultiLabelEvolveGCN(
        num_nodes=param["NUM_NODES"],
        node_features=param["NUM_REL_FEATURES"] + param["NUM_IRR_FEATURES"] + param["NUM_RED_FEATURES"],
        num_labels=param["NUM_LABELS"]
    ).to(DEVICE)

    optimizer_mlegcn = th.optim.Adam(model_mlegcn.parameters(), lr=param["LR_MLEGCN"])

    mlegcn = {'model': model_mlegcn,
              'optimizer': optimizer_mlegcn,
              'type': "mlegcn"}

    loss = th.nn.BCEWithLogitsLoss()

    models = {'tmf' : tmf,
              'mlegcn' : mlegcn,
              'tmf_dw' : tmf_dw,}

    return models, loss

def train_model(model, train_dataset, val_dataset, test_dataset, train_embeddings, val_embeddings, test_embeddings, loss_fn, params):
    model["model"].train()

    min_val_loss = float("inf")

    model_results = ModelResults()

    for epoch in tqdm(range(params["EPOCHS"])):
        loss = 0

        # training
        model["model"].train()
        model["optimizer"].zero_grad()

        for time, snapshot in enumerate(train_dataset):
            embedding = train_embeddings[time].to(DEVICE)
            snapshot = snapshot.to(DEVICE)

            if model["type"] == "tmf":
                y_hat = model["model"](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr)
            elif model["type"] == "tmf_dw":
                y_hat = model["model"](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, embedding)
            elif model["type"] == "mlegcn":
                y_hat = model["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            else:
                raise Exception(f'{model["type"]} model does not exist')

            loss += loss_fn(y_hat, snapshot.y)

        loss /= train_dataset.snapshot_count
        loss.backward()
        model["optimizer"].step()

        # training eval
        train_results = evaluate_model(model, train_dataset, train_embeddings, loss_fn, params)
        train_results.loss = loss.cpu().item()
        model_results.train_metrics.append_metrics(train_results)

        val_results = evaluate_model(model, val_dataset, val_embeddings, loss_fn, params)
        model_results.val_metrics.append_metrics(val_results)

        if val_results.loss < min_val_loss:
            test_results = evaluate_model(model, test_dataset, test_embeddings, loss_fn, params)
            model_results.test_metrics = test_results

            min_val_loss = val_results.loss

    return model_results

def evaluate_model(model, test_dataset, embeddings, loss_fn, param):
    model["model"].eval()

    total_loss = 0
    all_preds = []
    all_targets = []

    with th.no_grad():
        for time, snapshot in enumerate(test_dataset):
            embedding = embeddings[time].to(DEVICE)
            snapshot = snapshot.to(DEVICE)

            if model["type"] == "tmf":
                y_hat = model["model"](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr)
            elif model["type"] == "tmf_dw":
                y_hat = model["model"](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, embedding)
            elif model["type"] == "mlegcn":
                y_hat = model["model"](snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            else:
                raise Exception(f'{model["type"]} model does not exist')

            loss_tmf = loss_fn(y_hat, snapshot.y)
            total_loss += loss_tmf.cpu().item()

            # Apply sigmoid and threshold to get predictions
            preds = (th.sigmoid(y_hat) > param["THRESHOLD"]).float()

            all_preds.append(preds.cpu())
            all_targets.append(snapshot.y.cpu())

    loss = total_loss / test_dataset.snapshot_count

    # Concatenate all batches and compute metrics
    all_preds = th.cat(all_preds, dim=0).numpy()
    all_targets = th.cat(all_targets, dim=0).numpy()
    results = metrics(all_targets, all_preds)
    results.loss = loss

    return results

def experiment(param, datasets, display = True):
    # run singular repeat of experiment on both models with same dataset
    # setup models
    models, loss = initialize_models(param)
    model_names = ["Temporal MultiFix", "Temporal MultiFix Deepwalk", "MultiFix Evolve GCN"]

    train_dataset = datasets["train_data"]
    validation_dataset = datasets["validation_data"]
    test_dataset = datasets["test_data"]
    train_embedding = datasets["train_emb"]
    validation_embedding = datasets["validation_emb"]
    test_embedding = datasets["test_emb"]
    results = {}
    out = ""

    for model, name in zip(models.items(), model_names):
        repeat_results = []

        for r in range(param["REPEATS"]):
            model_results = train_model(model[1], train_dataset, validation_dataset, test_dataset, train_embedding, validation_embedding, test_embedding, loss, param)
            repeat_results.append(model_results)

        aggregated_metrics = ModelResults.aggregate_results(repeat_results)

        results[model[1]["type"]] = aggregated_metrics

        out += (f"\n{name}: \n"
               +str(aggregated_metrics.test_metrics))

    out += f'\n Dataset Inter Homophily:{datasets["inter_homophily"]:.4f}\n'
    # save to txt
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'{param["DATA_FILE"]}_alpha{param["ALPHA"]}.txt'
    filepath = os.path.join(param["EXPERIMENT_PATH"], filename)

    with open(filepath, "w") as file:
        file.write(out + str(param))

    print(out)

    plotting(param, results, datasets, False)
    plotting(param, results, datasets, True)

    return results

def plotting(param, results, datasets, zoom = False):
    x = range(param["EPOCHS"])
    metric_names = ['BCE Loss', 'F1 Macro', 'F1 Micro', 'AP Macro', 'AUC ROC']
    file_names = ['loss-curve', 'f1-macro-curve', 'f1-micro-curve', 'ap-macro-curve', 'auc-roc-curve']

    train_datas_tmf = results["tmf"].train_metrics.metric_list()
    validation_datas_tmf = results["tmf"].val_metrics.metric_list()

    train_datas_tmf_dw = results["tmf_dw"].train_metrics.metric_list()
    validation_datas_tmf_dw = results["tmf_dw"].val_metrics.metric_list()

    train_datas_mlegcn = results["mlegcn"].train_metrics.metric_list()
    validation_datas_mlegcn = results["mlegcn"].val_metrics.metric_list()

    for i in range(len(metric_names)):

        title = metric_names[i]
        imgname = file_names[i]


        # plot 1
        plt.figure(figsize=(8, 5), dpi=600)

        train_loss_mean = np.mean(train_datas_tmf[i], axis=0)
        train_loss_std = np.std(train_datas_tmf[i], axis=0)
        plt.plot(x, train_loss_mean, label='Temporal MultiFix (Train)', color='tab:red')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color='tab:red', alpha=0.3,
                         label='_nolegend_')

        train_loss_mean = np.mean(train_datas_tmf_dw[i], axis=0)
        train_loss_std = np.std(train_datas_tmf_dw[i], axis=0)
        plt.plot(x, train_loss_mean, label='Temporal MultiFix DW (Train)', color='tab:blue')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color='tab:blue',
                         alpha=0.3, label='_nolegend_')

        train_loss_mean = np.mean(train_datas_mlegcn[i], axis=0)
        train_loss_std = np.std(train_datas_mlegcn[i], axis=0)
        plt.plot(x, train_loss_mean, label='MultiLabel Evolve GCN (Train)', color='tab:green')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color='tab:green',
                         alpha=0.3, label='_nolegend_')

        train_loss_mean = np.mean(validation_datas_tmf[i], axis=0)
        train_loss_std = np.std(validation_datas_tmf[i], axis=0)
        plt.plot(x, train_loss_mean, '--', label='Temporal MultiFix (Val)', color='tab:pink')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                         color='tab:pink', alpha=0.3, label='_nolegend_')

        train_loss_mean = np.mean(validation_datas_tmf_dw[i], axis=0)
        train_loss_std = np.std(validation_datas_tmf_dw[i], axis=0)
        plt.plot(x, train_loss_mean, '--', label='Temporal MultiFix DW (Val)', color='tab:cyan')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                         color='tab:cyan', alpha=0.3, label='_nolegend_')

        train_loss_mean = np.mean(validation_datas_mlegcn[i], axis=0)
        train_loss_std = np.std(validation_datas_mlegcn[i], axis=0)
        plt.plot(x, train_loss_mean, '--', label='MultiLabel Evolve GCN (Val)', color='tab:olive')
        plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                         color='tab:olive', alpha=0.3, label='_nolegend_')

        plt.xlabel("Epoch")
        plt.ylabel(f"{title} (mean ± std)")
        plt.title(f'alpha={param["ALPHA"]}, inter homophily={datasets["inter_homophily"]:.2f}')
        plt.grid(True)


        plt.legend(loc='best', ncols=2)

        plt.tight_layout()
        if zoom:
            plt.ylim(0, 1)
            filename = f'{param["IMAGE_FILE"]}_alpha{param["ALPHA"]}_{imgname}-zoom.png'
        else:
            filename = f'{param["IMAGE_FILE"]}_alpha{param["ALPHA"]}_{imgname}.png'
        filepath = os.path.join(param["EXPERIMENT_PATH"], filename)
        plt.savefig(filepath)
        plt.close()

def experiment_main(param):
    # setup export, establish folder to store all results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    param["EXPERIMENT_PATH"] = os.path.join(param["EXPERIMENT_PATH"], timestamp)
    os.makedirs(param["EXPERIMENT_PATH"], exist_ok=True)

    config = TemporalMultiLabelGeneratorConfig(m_rel=param["NUM_REL_FEATURES"],
                                                m_irr=param["NUM_IRR_FEATURES"],
                                                m_red=param["NUM_RED_FEATURES"],
                                                q=param["NUM_LABELS"],
                                                N=param["NUM_NODES"],
                                                max_r=0.8,
                                                min_r=((param["NUM_LABELS"] / 10) + 1) / param["NUM_LABELS"],
                                                mu=0,
                                                b=0.05,
                                                alpha=param["ALPHA"],
                                                theta=np.pi / -(param["NUM_TIMESTEPS"]//-2), #ceiling division
                                                horizon=param["NUM_TIMESTEPS"],
                                                sphere_sampling='polar',
                                                data_sampling='global',
                                                rotation_reference='data',
                                                )
    
    dataset_loader = DatasetLoader(config, param["EMBEDDING_DIM"], param["FILENAME"], param["BASEFILE"])
    
    # alphas = [5, 4, 3, 2, 1, 0]
    # alphas = [6, 7, 8, 9, 10]
    alphas = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    for alpha in alphas[::-1]:
        # experiment 
        param["ALPHA"] = alpha
        dataset_loader.generator.alpha = alpha
        datasets = load_data(dataset_loader, param)
        
        # experiment_repeats(param, datasets)

        experiment(param, datasets)



if __name__ == "__main__":
    print(f'{DEVICE} available')
    experiment_main(PARAM)

