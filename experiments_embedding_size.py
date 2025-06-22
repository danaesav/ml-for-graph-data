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
        'NUM_IRR_FEATURES' : 10,
        'NUM_RED_FEATURES' : 0,
        'NUM_LABELS' : 20,  # q = number of hyperspheres
        'NUM_TIMESTEPS' : 30,  # horizon
        'EPOCHS' : 500,
        'LR_MLEGCN': 2e-2,
        'LR_TMF': 8e-3,
        'THRESHOLD' : 0.5,  # for classification
        'REPEATS' : 3, 
        'ALPHA' : 5,
        'EMBEDDING_DIM' : [16], #[16,32,64], 
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

    models = {}
    for embed in param["EMBEDDING_DIM"]:

        # Temporal MultiFix with Deepwalk embeddings
        model_tmf_dw = TemporalMultiFix(
            input_dim=param["NUM_REL_FEATURES"] + param["NUM_IRR_FEATURES"] + param["NUM_RED_FEATURES"],
            num_of_nodes=param["NUM_NODES"],
            output_dim=param["NUM_LABELS"],
            dw_dim=embed,).to(DEVICE)

        optimizer_tmf_dw = th.optim.Adam(model_tmf_dw.parameters(), lr=param["LR_TMF"])

        tmf_dw = {'model':model_tmf_dw,
                'optimizer':optimizer_tmf_dw,
                'type': f"tmf_dw{embed}"}
        
        models[f"tmf_dw{embed}"] = tmf_dw


    loss = th.nn.BCEWithLogitsLoss()

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
            elif model["type"].startswith("tmf_dw"):
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
            elif model["type"].startswith("tmf_dw"):
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
    model_names = [f"Temporal MultiFix Deepwalk (EmbDim={embed})" for embed in param["EMBEDDING_DIM"]]

    train_dataset = [dataset_i["train_data"] for dataset_i in datasets]
    validation_dataset = [dataset_i["validation_data"] for dataset_i in datasets]
    test_dataset = [dataset_i["test_data"] for dataset_i in datasets]
    train_embedding = [dataset_i["train_emb"] for dataset_i in datasets]
    validation_embedding = [dataset_i["validation_emb"] for dataset_i in datasets]
    test_embedding = [dataset_i["test_emb"] for dataset_i in datasets]
    results = {}
    out = ""

    for i, (model, name) in enumerate(zip(models.items(), model_names)):
        repeat_results = []

        for r in range(param["REPEATS"]):
            model_results = train_model(model[1], train_dataset[i], validation_dataset[i], test_dataset[i], train_embedding[i], validation_embedding[i], test_embedding[i], loss, param)
            repeat_results.append(model_results)

        aggregated_metrics = ModelResults.aggregate_results(repeat_results)

        results[model[1]["type"]] = aggregated_metrics

        out += (f"\n{name}: \n"
               +str(aggregated_metrics.test_metrics))

    out += f'\n Dataset Inter Homophily:{np.mean([dataset_i["inter_homophily"] for dataset_i in datasets]):.4f}\n'
    # save to txt
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'{param["DATA_FILE"]}_alpha{param["ALPHA"]}.txt'
    filepath = os.path.join(param["EXPERIMENT_PATH"], filename)

    with open(filepath, "w") as file:
        file.write(out + str(param))

    print(out)

    # print('testing\n\n')
    # print(results)
    # print()
    # print(datasets)
    
    plotting(param, results, datasets, False)
    plotting(param, results, datasets, True)

    return results

def plotting(param, results, datasets, zoom = False):
    x = range(param["EPOCHS"])
    metric_names = ['BCE Loss', 'F1 Macro', 'F1 Micro', 'AP Macro', 'AUC ROC']
    file_names = ['loss-curve', 'f1-macro-curve', 'f1-micro-curve', 'ap-macro-curve', 'auc-roc-curve']

    for i in range(len(metric_names)):

        title = metric_names[i]
        imgname = file_names[i]


        # plot 1
        plt.figure(figsize=(8, 5), dpi=600)

        colors = [("tab:blue", "tab:cyan"), ("tab:red","tab:pink"), ("tab:green", "tab:olive")]
        plot_names = [(f'Embedding size = {p} (Train)', f'Embedding size = {p} (Val)') for p in param["EMBEDDING_DIM"]]

        for j, (_, model) in enumerate(results.items()):
            train_datas_tmf_dw = model.train_metrics.metric_list()
            validation_datas_tmf_dw = model.val_metrics.metric_list()

            train_loss_mean = np.mean(train_datas_tmf_dw[i], axis=0)
            train_loss_std = np.std(train_datas_tmf_dw[i], axis=0)
            plt.plot(x, train_loss_mean, label=plot_names[j][0], color=colors[j][0])
            plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, color=colors[j][0],
                            alpha=0.3, label='_nolegend_')

            train_loss_mean = np.mean(validation_datas_tmf_dw[i], axis=0)
            train_loss_std = np.std(validation_datas_tmf_dw[i], axis=0)
            plt.plot(x, train_loss_mean, '--', label=plot_names[j][1], color=colors[j][1])
            plt.fill_between(x, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                            color=colors[j][1], alpha=0.3, label='_nolegend_')


        plt.xlabel("Epoch")
        plt.ylabel(f"{title} (mean ± std)")
        plt.title(f'Temporal MultiFix DW, alpha={param["ALPHA"]}, inter homophily={np.mean([dataset_i["inter_homophily"] for dataset_i in datasets]):.2f}')
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
    
    
    alphas = [3]
    for alpha in alphas[::-1]:
        datasets = []
        # experiment 
        param["ALPHA"] = alpha
        for embed in param["EMBEDDING_DIM"]:
            dataset_loader = DatasetLoader(config, embed, filename=param["FILENAME"], load=param["BASEFILE"])
            dataset_loader.generator.alpha = alpha
            datasets.append(load_data(dataset_loader, param))
        
        # experiment_repeats(param, datasets)

        experiment(param, datasets)



if __name__ == "__main__":
    print(f'{DEVICE} available')
    experiment_main(PARAM)

