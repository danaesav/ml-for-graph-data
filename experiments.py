import numpy as np
import torch as th
import os

from tqdm import tqdm
from torch_geometric_temporal import temporal_signal_split
from matplotlib import pyplot as plt
from datetime import datetime

from generator.temporal_multi_label_generator import TemporalMultiLabelGeneratorConfig
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
        'NUM_TIMESTEPS' : 15,  # horizon
        'EPOCHS' : 200,
        'LR': 1e-2,
        'THRESHOLD' : 0.5,  # for classification
        'REPEATS' : 5,
        'ALPHA' : 5,
        'EMBEDDING_DIM' : 128,
        'TRAIN_RATIO' : 0.8,
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
    return (*temporal_signal_split(dataset, train_ratio=param['TRAIN_RATIO']),
            *temporal_signal_split_list(embeddings, train_ratio=param['TRAIN_RATIO']),
            inter_homophily,
            intra_homophily)


def initialize_models(param):
    # deepwalk_embeddings = None #shape(TIME, NUM_NODES, NUM_FEATURES)

    # Temporal MultiFix with Deepwalk embeddings
    model_tmf_dw = TemporalMultiFix(
        input_dim=param['NUM_REL_FEATURES'] + param['NUM_IRR_FEATURES'] + param['NUM_RED_FEATURES'],
        num_of_nodes=param['NUM_NODES'],
        output_dim=param['NUM_LABELS'],
        dw_dim=param['EMBEDDING_DIM'],).to(DEVICE)

    optimizer_tmf_dw = th.optim.Adam(model_tmf_dw.parameters(), lr=param['LR'])

    tmf_dw = {'model':model_tmf_dw,
              'optimizer':optimizer_tmf_dw}


    # Temporal MultiFix without Deepwalk Embeddings
    model_tmf = TemporalMultiFix(
        input_dim=param['NUM_REL_FEATURES'] + param['NUM_IRR_FEATURES'] + param['NUM_RED_FEATURES'],
        num_of_nodes=param['NUM_NODES'],
        output_dim=param['NUM_LABELS']).to(DEVICE)

    optimizer_tmf = th.optim.Adam(model_tmf.parameters(), lr=param['LR'])

    tmf = {'model': model_tmf,
              'optimizer': optimizer_tmf}


    # MultiLabel EvolveGCN
    model_mlegcn = MultiLabelEvolveGCN(
        num_nodes=param['NUM_NODES'],
        node_features=param['NUM_REL_FEATURES'] + param['NUM_IRR_FEATURES'] + param['NUM_RED_FEATURES'],
        num_labels=param['NUM_LABELS']
    ).to(DEVICE)

    optimizer_mlegcn = th.optim.Adam(model_mlegcn.parameters(), lr=param['LR'])

    mlegcn = {'model': model_mlegcn,
              'optimizer': optimizer_mlegcn}

    loss = th.nn.BCEWithLogitsLoss()

    models = {'tmf' : tmf,
              'mlegcn' : mlegcn,
              'tmf_dw' : tmf_dw,}

    return models, loss


def train(models, train_dataset, embeddings, loss_fn, params):
    models['tmf']['model'].train()
    models['tmf_dw']['model'].train()
    models['mlegcn']['model'].train()

    losses = {'tmf' : [],
              'tmf_dw' : [],
              'mlegcn' : [],}
    
    f1_micro = {'tmf':[],'tmf_dw':[],'mlegcn':[]}
    f1_macro = {'tmf': [],'tmf_dw': [],'mlegcn': []}
    ap_macro = {'tmf': [],'tmf_dw': [],'mlegcn': []}
    auc_roc = {'tmf': [],'tmf_dw': [],'mlegcn': []}

    results = {'train_loss':losses,
               'f1_micro':f1_micro,
               'f1_macro':f1_macro,
               'ap_macro':ap_macro,
               'auc_roc':auc_roc}

    for epoch in tqdm(range(params['EPOCHS'])):
        total_loss_tmf = 0
        total_loss_tmf_dw = 0
        total_loss_mlegcn = 0

        #training 
        for time, snapshot in enumerate(train_dataset):
            embedding = embeddings[time].to(DEVICE)
            snapshot = snapshot.to(DEVICE)

            models['tmf']['optimizer'].zero_grad()
            models['tmf_dw']['optimizer'].zero_grad()
            models['mlegcn']['optimizer'].zero_grad()

            y_hat_tmf = models['tmf']['model'](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr)
            y_hat_tmf_dw = models['tmf_dw']['model'](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, embedding)
            y_hat_mlegcn = models['mlegcn']['model'](snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            loss_tmf = loss_fn(y_hat_tmf, snapshot.y)
            loss_tmf_dw = loss_fn(y_hat_tmf_dw, snapshot.y)
            loss_mlegcn = loss_fn(y_hat_mlegcn, snapshot.y)

            loss_tmf.backward()
            loss_tmf_dw.backward()
            loss_mlegcn.backward()

            models['tmf']['optimizer'].step()
            models['tmf_dw']['optimizer'].step()
            models['mlegcn']['optimizer'].step()

            total_loss_tmf += loss_tmf.cpu().item()
            total_loss_tmf_dw += loss_tmf_dw.cpu().item()
            total_loss_mlegcn += loss_mlegcn.cpu().item()

        results['train_loss']['tmf'].append(total_loss_tmf / train_dataset.snapshot_count)
        results['train_loss']['tmf_dw'].append(total_loss_tmf_dw / train_dataset.snapshot_count)
        results['train_loss']['mlegcn'].append(total_loss_mlegcn / train_dataset.snapshot_count)

        #training eval
        eval_results = evaluate(models, train_dataset, embeddings, loss_fn, params)

        results['f1_micro']['tmf'].append(eval_results['f1_micro']['tmf'])
        results['f1_micro']['tmf_dw'].append(eval_results['f1_micro']['tmf_dw'])
        results['f1_micro']['mlegcn'].append(eval_results['f1_micro']['mlegcn'])

        results['f1_macro']['tmf'].append(eval_results['f1_macro']['tmf'])
        results['f1_macro']['tmf_dw'].append(eval_results['f1_macro']['tmf_dw'])
        results['f1_macro']['mlegcn'].append(eval_results['f1_macro']['mlegcn'])

        results['ap_macro']['tmf'].append(eval_results['ap_macro']['tmf'])
        results['ap_macro']['tmf_dw'].append(eval_results['ap_macro']['tmf_dw'])
        results['ap_macro']['mlegcn'].append(eval_results['ap_macro']['mlegcn'])

        results['auc_roc']['tmf'].append(eval_results['auc_roc']['tmf'])
        results['auc_roc']['tmf_dw'].append(eval_results['auc_roc']['tmf_dw'])
        results['auc_roc']['mlegcn'].append(eval_results['auc_roc']['mlegcn'])

    return results


def evaluate(models, test_dataset, embeddings, loss_fn, param):

    #setup return container
    test_loss = {'tmf':0,'tmf_dw':0,'mlegcn':0,}
    f1_micro = {'tmf':0,'tmf_dw':0,'mlegcn':0}
    f1_macro = {'tmf': 0,'tmf_dw': 0,'mlegcn': 0}
    ap_macro = {'tmf': 0,'tmf_dw': 0,'mlegcn': 0}
    auc_roc = {'tmf': 0,'tmf_dw': 0,'mlegcn': 0}

    results = {'test_loss':test_loss,
               'f1_micro':f1_micro,
               'f1_macro':f1_macro,
               'ap_macro':ap_macro,
               'auc_roc':auc_roc}

    # evaluate
    models['tmf']['model'].eval()
    models['tmf_dw']['model'].eval()
    models['mlegcn']['model'].eval()

    total_loss_tmf = 0
    total_loss_tmf_dw = 0
    total_loss_mlegcn = 0

    all_preds_tmf = []
    all_preds_tmf_dw = []
    all_preds_mlegcn = []

    all_targets = []

    with th.no_grad():
        for time, snapshot in enumerate(test_dataset):
            embedding = embeddings[time].to(DEVICE)
            snapshot = snapshot.to(DEVICE)

            y_hat_tmf = models['tmf']['model'](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr)
            y_hat_tmf_dw = models['tmf_dw']['model'](snapshot.x, snapshot.y, snapshot.edge_index, snapshot.edge_attr, embedding)
            y_hat_mlegcn = models['mlegcn']['model'](snapshot.x, snapshot.edge_index, snapshot.edge_attr)

            loss_tmf = loss_fn(y_hat_tmf, snapshot.y)
            loss_tmf_dw = loss_fn(y_hat_tmf_dw, snapshot.y)
            loss_mlegcn = loss_fn(y_hat_mlegcn, snapshot.y)

            total_loss_tmf += loss_tmf.cpu().item()
            total_loss_tmf_dw += loss_tmf_dw.cpu().item()
            total_loss_mlegcn += loss_mlegcn.cpu().item()

            # Apply sigmoid and threshold to get predictions
            preds_tmf = (th.sigmoid(y_hat_tmf) > param['THRESHOLD']).float()
            preds_tmf_dw = (th.sigmoid(y_hat_tmf_dw) > param['THRESHOLD']).float()
            preds_mlegcn = (th.sigmoid(y_hat_mlegcn) > param['THRESHOLD']).float()

            all_preds_tmf.append(preds_tmf.cpu())
            all_preds_tmf_dw.append(preds_tmf_dw.cpu())
            all_preds_mlegcn.append(preds_mlegcn.cpu())

            all_targets.append(snapshot.y.cpu())

    results['test_loss']['tmf'] = total_loss_tmf / test_dataset.snapshot_count
    results['test_loss']['tmf_dw'] = total_loss_tmf_dw / test_dataset.snapshot_count
    results['test_loss']['mlegcn'] = total_loss_mlegcn / test_dataset.snapshot_count

    # Concatenate all batches and compute metrics
    all_preds_tmf = th.cat(all_preds_tmf, dim=0).numpy()
    all_preds_tmf_dw = th.cat(all_preds_tmf_dw, dim=0).numpy()
    all_preds_mlegcn = th.cat(all_preds_mlegcn, dim=0).numpy()

    all_targets = th.cat(all_targets, dim=0).numpy()

    results['f1_macro']['tmf'], results['f1_micro']['tmf'], results['auc_roc']['tmf'], results['ap_macro']['tmf'] = metrics(all_targets, all_preds_tmf)
    results['f1_macro']['tmf_dw'], results['f1_micro']['tmf_dw'], results['auc_roc']['tmf_dw'], results['ap_macro']['tmf_dw'] = metrics(all_targets, all_preds_tmf_dw)
    results['f1_macro']['mlegcn'], results['f1_micro']['mlegcn'], results['auc_roc']['mlegcn'], results['ap_macro']['mlegcn'] = metrics(all_targets, all_preds_mlegcn)

    return results



def experiment_single_run(param, datasets, display = True):
    # run singular repeat of experiment on both models with same dataset

    # setup experiment
    results = {
        'train-loss':None,
        'train-f1 macro':None,
        'train-f1 micro':None,
        'train-ap macro':None,
        'train-auc roc':None,
        'test-loss':None,
        'test-f1 macro':None,
        'test-f1 micro':None,
        'test-ap macro':None,
        'test-auc roc':None,
    }

    train_dataset = datasets['train_data']
    test_dataset = datasets['test_data']
    train_embedding = datasets['train_emb']
    test_embedding = datasets['test_emb']

    # setup models
    models, loss = initialize_models(param)

    # train models
    train_results = train(models, train_dataset, train_embedding, loss, param)
    results['train-loss'] = train_results['train_loss']
    results['train-f1 macro'] = train_results['f1_macro']
    results['train-f1 micro'] = train_results['f1_micro']
    results['train-ap macro'] = train_results['ap_macro']
    results['train-auc roc'] = train_results['auc_roc']

    # test models
    test_results = evaluate(models, test_dataset, test_embedding, loss, param)

    results['test-loss'] = test_results['test_loss']
    results['test-f1 macro'] = test_results['f1_macro']
    results['test-f1 micro'] = test_results['f1_micro']
    results['test-ap macro'] = test_results['ap_macro']
    results['test-auc roc'] = test_results['auc_roc']

    if display:
        out  = (f"Temporal MultiFix: train-loss:{results['train-loss']['tmf'][-1]:.4f}, "
                + f"test-loss:{results['test-loss']['tmf']:.4f}, "
                + f"test-f1-macro:{results['test-f1 macro']['tmf']:.4f}, "
                + f"test-f1-micro:{results['test-f1 micro']['tmf']:.4f}, "
                + f"test-AP-macro:{results['test-ap macro']['tmf']:.4f}, "
                + f"test-AUC-ROC:{results['test-auc roc']['tmf']:.4f} \n"
                + f"Temporal MultiFix Deepwalk: train-loss:{results['train-loss']['tmf_dw'][-1]:.4f}, "
                + f"test-loss:{results['test-loss']['tmf_dw']:.4f}, "
                + f"test-f1-macro:{results['test-f1 macro']['tmf_dw']:.4f}, "
                + f"test-f1-micro:{results['test-f1 micro']['tmf_dw']:.4f}, "
                + f"test-AP-macro:{results['test-ap macro']['tmf_dw']:.4f}, "
                + f"test-AUC-ROC:{results['test-auc roc']['tmf_dw']:.4f} \n"
                + f"MultiFix Evolve GCN: train-loss:{results['train-loss']['mlegcn'][-1]:.4f}, "
                + f"test-loss:{results['test-loss']['mlegcn']:.4f}, "
                + f"test-f1-macro:{results['test-f1 macro']['mlegcn']:.4f}, "
                + f"test-f1-micro:{results['test-f1 micro']['mlegcn']:.4f}, "
                + f"test-AP-macro:{results['test-ap macro']['mlegcn']:.4f}, "
                + f"test-AUC-ROC:{results['test-auc roc']['mlegcn']:.4f} \n"
                )

        print(out)

    return results



def experiment_repeats(param, datasets, display=True):

    train_loss_tmf = []
    train_f1_macro_tmf = []
    train_f1_micro_tmf = []
    train_ap_macro_tmf = []
    train_auc_roc_tmf = []
    test_loss_tmf = []
    test_f1_macro_tmf = []
    test_f1_micro_tmf = []
    test_ap_macro_tmf = []
    test_auc_roc_tmf = []

    inter_homophily = datasets['inter_homophily']

    train_loss_tmf_dw = []
    train_f1_macro_tmf_dw = []
    train_f1_micro_tmf_dw = []
    train_ap_macro_tmf_dw = []
    train_auc_roc_tmf_dw = []
    test_loss_tmf_dw = []
    test_f1_macro_tmf_dw = []
    test_f1_micro_tmf_dw = []
    test_ap_macro_tmf_dw = []
    test_auc_roc_tmf_dw = []

    train_loss_mlegcn = []
    train_f1_macro_mlegcn = []
    train_f1_micro_mlegcn = []
    train_ap_macro_mlegcn = []
    train_auc_roc_mlegcn = []
    test_loss_mlegcn = []
    test_f1_macro_mlegcn = []
    test_f1_micro_mlegcn = []
    test_ap_macro_mlegcn = []
    test_auc_roc_mlegcn = []

    for r in range(param['REPEATS']):
        print(f'Repeat: {r}')

        results = experiment_single_run(param, datasets, display)

        # inter_homophily.append(results['inter-homophily'])

        train_loss_tmf.append(results['train-loss']['tmf'])
        train_f1_macro_tmf.append(results['train-f1 macro']['tmf'])
        train_f1_micro_tmf.append(results['train-f1 micro']['tmf'])
        train_ap_macro_tmf.append(results['train-ap macro']['tmf'])
        train_auc_roc_tmf.append(results['train-auc roc']['tmf'])
        test_loss_tmf.append(results['test-loss']['tmf'])
        test_f1_macro_tmf.append(results['test-f1 macro']['tmf'])
        test_f1_micro_tmf.append(results['test-f1 micro']['tmf'])
        test_ap_macro_tmf.append(results['test-ap macro']['tmf'])
        test_auc_roc_tmf.append(results['test-auc roc']['tmf'])

        train_loss_tmf_dw.append(results['train-loss']['tmf_dw'])
        train_f1_macro_tmf_dw.append(results['train-f1 macro']['tmf_dw'])
        train_f1_micro_tmf_dw.append(results['train-f1 micro']['tmf_dw'])
        train_ap_macro_tmf_dw.append(results['train-ap macro']['tmf_dw'])
        train_auc_roc_tmf_dw.append(results['train-auc roc']['tmf_dw'])
        test_loss_tmf_dw.append(results['test-loss']['tmf_dw'])
        test_f1_macro_tmf_dw.append(results['test-f1 macro']['tmf_dw'])
        test_f1_micro_tmf_dw.append(results['test-f1 micro']['tmf_dw'])
        test_ap_macro_tmf_dw.append(results['test-ap macro']['tmf_dw'])
        test_auc_roc_tmf_dw.append(results['test-auc roc']['tmf_dw'])

        train_loss_mlegcn.append(results['train-loss']['mlegcn'])
        train_f1_macro_mlegcn.append(results['train-f1 macro']['mlegcn'])
        train_f1_micro_mlegcn.append(results['train-f1 micro']['mlegcn'])
        train_ap_macro_mlegcn.append(results['train-ap macro']['mlegcn'])
        train_auc_roc_mlegcn.append(results['train-auc roc']['mlegcn'])
        test_loss_mlegcn.append(results['test-loss']['mlegcn'])
        test_f1_macro_mlegcn.append(results['test-f1 macro']['mlegcn'])
        test_f1_micro_mlegcn.append(results['test-f1 micro']['mlegcn'])
        test_ap_macro_mlegcn.append(results['test-ap macro']['mlegcn'])
        test_auc_roc_mlegcn.append(results['test-auc roc']['mlegcn'])


    train_loss_tmf_mean = np.mean(np.array(train_loss_tmf)[:, -1])
    train_loss_tmf_std = np.std(np.array(train_loss_tmf)[:, -1])
    test_loss_tmf_mean = np.mean(test_loss_tmf)
    test_loss_tmf_std = np.std(test_loss_tmf)
    test_f1_macro_tmf_mean = np.mean(test_f1_macro_tmf)
    test_f1_macro_tmf_std = np.std(test_f1_macro_tmf)
    test_f1_micro_tmf_mean = np.mean(test_f1_micro_tmf)
    test_f1_micro_tmf_std = np.std(test_f1_micro_tmf)
    test_ap_macro_tmf_mean = np.mean(test_ap_macro_tmf)
    test_ap_macro_tmf_std = np.std(test_ap_macro_tmf)
    test_auc_roc_tmf_mean = np.mean(test_auc_roc_tmf)
    test_auc_roc_tmf_std = np.std(test_auc_roc_tmf)

    train_loss_tmf_dw_mean = np.mean(np.array(train_loss_tmf_dw)[:, -1])
    train_loss_tmf_dw_std = np.std(np.array(train_loss_tmf_dw)[:, -1])
    test_loss_tmf_dw_mean = np.mean(test_loss_tmf_dw)
    test_loss_tmf_dw_std = np.std(test_loss_tmf_dw)
    test_f1_macro_tmf_dw_mean = np.mean(test_f1_macro_tmf_dw)
    test_f1_macro_tmf_dw_std = np.std(test_f1_macro_tmf_dw)
    test_f1_micro_tmf_dw_mean = np.mean(test_f1_micro_tmf_dw)
    test_f1_micro_tmf_dw_std = np.std(test_f1_micro_tmf_dw)
    test_ap_macro_tmf_dw_mean = np.mean(test_ap_macro_tmf_dw)
    test_ap_macro_tmf_dw_std = np.std(test_ap_macro_tmf_dw)
    test_auc_roc_tmf_dw_mean = np.mean(test_auc_roc_tmf_dw)
    test_auc_roc_tmf_dw_std = np.std(test_auc_roc_tmf_dw)

    train_loss_mlegcn_mean = np.mean(np.array(train_loss_mlegcn)[:, -1])
    train_loss_mlegcn_std = np.std(np.array(train_loss_mlegcn)[:, -1])
    test_loss_mlegcn_mean = np.mean(test_loss_mlegcn)
    test_loss_mlegcn_std = np.std(test_loss_mlegcn)
    test_f1_macro_mlegcn_mean = np.mean(test_f1_macro_mlegcn)
    test_f1_macro_mlegcn_std = np.std(test_f1_macro_mlegcn)
    test_f1_micro_mlegcn_mean = np.mean(test_f1_micro_mlegcn)
    test_f1_micro_mlegcn_std = np.std(test_f1_micro_mlegcn)
    test_ap_macro_mlegcn_mean = np.mean(test_ap_macro_mlegcn)
    test_ap_macro_mlegcn_std = np.std(test_ap_macro_mlegcn)
    test_auc_roc_mlegcn_mean = np.mean(test_auc_roc_mlegcn)
    test_auc_roc_mlegcn_std = np.std(test_auc_roc_mlegcn)

    if display:
        out = (f"\nTemporal MultiFix: \ntrain-loss:{train_loss_tmf_mean:.4f}+-{train_loss_tmf_std:.4f}\n"
               + f"test-loss:{test_loss_tmf_mean:.4f}+-{test_loss_tmf_std:.4f}\n"
               + f"test-f1-macro:{test_f1_macro_tmf_mean:.4f}+-{test_f1_macro_tmf_std:.4f}\n"
               + f"test-f1-micro:{test_f1_micro_tmf_mean:.4f}+-{test_f1_micro_tmf_std:.4f}\n"
               + f"test-AP-macro:{test_ap_macro_tmf_mean:.4f}+-{test_ap_macro_tmf_std:.4f}\n"
               + f"test-AUC-ROC:{test_auc_roc_tmf_mean:.4f}+-{test_auc_roc_tmf_std:.4f}\n"
               + f"\nTemporal MultiFix Deepwalk: \ntrain-loss:{train_loss_tmf_dw_mean:.4f}+-{train_loss_tmf_dw_std:.4f}\n"
               + f"test-loss:{test_loss_tmf_dw_mean:.4f}+-{test_loss_tmf_dw_std:.4f}\n"
               + f"test-f1-macro:{test_f1_macro_tmf_dw_mean:.4f}+-{test_f1_macro_tmf_dw_std:.4f}\n"
               + f"test-f1-micro:{test_f1_micro_tmf_dw_mean:.4f}+-{test_f1_micro_tmf_dw_std:.4f}\n"
               + f"test-AP-macro:{test_ap_macro_tmf_dw_mean:.4f}+-{test_ap_macro_tmf_dw_std:.4f}\n"
               + f"test-AUC-ROC:{test_auc_roc_tmf_dw_mean:.4f}+-{test_auc_roc_tmf_dw_std:.4f}\n"
               + f"\nMulti-Label Evolve GCN: \ntrain-loss:{train_loss_mlegcn_mean:.4f}+-{train_loss_mlegcn_std:.4f}\n"
               + f"test-loss:{test_loss_mlegcn_mean:.4f}+-{test_loss_mlegcn_std:.4f}\n"
               + f"test-f1-macro:{test_f1_macro_mlegcn_mean:.4f}+-{test_f1_macro_mlegcn_std:.4f}\n"
               + f"test-f1-micro:{test_f1_micro_mlegcn_mean:.4f}+-{test_f1_micro_mlegcn_std:.4f}\n"
               + f"test-AP-macro:{test_ap_macro_mlegcn_mean:.4f}+-{test_ap_macro_mlegcn_std:.4f}\n"
               + f"test-AUC-ROC:{test_auc_roc_mlegcn_mean:.4f}+-{test_auc_roc_mlegcn_std:.4f}\n"
               + f"\n Dataset Inter Homophily:{inter_homophily:.4f}\n"
               )

        print(out)

        # save to txt
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{param['DATA_FILE']}_alpha{param['ALPHA']}.txt"
        filepath = os.path.join(param['EXPERIMENT_PATH'], filename)

        with open(filepath, "w") as file:
            file.write(out + str(param))

        #plot training metrics curves over epochs
        curves = [(train_loss_tmf, train_loss_tmf_dw, train_loss_mlegcn, 'Training Loss', 'loss-curve'),
                  (train_f1_macro_tmf, train_f1_macro_tmf_dw, train_f1_macro_mlegcn, 'Training F1 Macro', 'f1-macro-curve'),
                  (train_f1_micro_tmf, train_f1_micro_tmf_dw, train_f1_micro_mlegcn, 'Training F1 Micro', 'f1-micro-curve'),
                  (train_ap_macro_tmf, train_ap_macro_tmf_dw, train_ap_macro_mlegcn, 'Training AP Macro', 'ap-macro-curve'),
                  (train_auc_roc_tmf, train_auc_roc_tmf_dw, train_auc_roc_mlegcn, 'Training AUC ROC', 'auc-roc-curve')
                  ]

        for data_tmf, data_tmf_dw, data_mlegcn, title, imgname in curves:

            #plot 1
            x = range(param['EPOCHS'])
            y_train_tmf = np.mean(np.array(data_tmf), axis=0) #size(R, E)
            y_train_tmf_std = np.std(np.array(data_tmf), axis=0)
            y_train_tmf_dw = np.mean(np.array(data_tmf_dw), axis=0)  # size(R, E)
            y_train_tmf_dw_std = np.std(np.array(data_tmf_dw), axis=0)
            y_train_mlegcn = np.mean(np.array(data_mlegcn), axis=0)  # size(R, E)
            y_train_mlegcn_std = np.std(np.array(data_mlegcn), axis=0)

            plt.figure(figsize=(8, 5))
            plt.errorbar(x, y_train_tmf, yerr=y_train_tmf_std, fmt='o-', capsize=5, label='Temporal MultiFix', color='red')
            plt.errorbar(x, y_train_tmf_dw, yerr=y_train_tmf_dw_std, fmt='o-', capsize=5, label='Temporal MultiFix DW', color='blue')
            plt.errorbar(x, y_train_mlegcn, yerr=y_train_mlegcn_std, fmt='o-', capsize=5, label='MultiLabel Evolve GCN', color='green')

            plt.xlabel("Epoch")
            plt.ylabel(f"{title} (mean ± std)")
            plt.title(f'alpha={param['ALPHA']}, inter homophily={inter_homophily:.2f}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            filename = f"{param['IMAGE_FILE']}_{imgname}.png"
            filepath = os.path.join(param['EXPERIMENT_PATH'], filename)
            plt.savefig(filepath)
            plt.close()
            
            # plot 2
            x = range(param['EPOCHS'])
            y_train_tmf = np.mean(np.array(train_loss_tmf), axis=0) #size(R, E)
            y_train_tmf_std = np.std(np.array(train_loss_tmf), axis=0)
            y_train_tmf_dw = np.mean(np.array(train_loss_tmf_dw), axis=0)  # size(R, E)
            y_train_tmf_dw_std = np.std(np.array(train_loss_tmf_dw), axis=0)
            y_train_mlegcn = np.mean(np.array(train_loss_mlegcn), axis=0)  # size(R, E)
            y_train_mlegcn_std = np.std(np.array(train_loss_mlegcn), axis=0)

            plt.figure(figsize=(8, 5))
            plt.plot(x, y_train_tmf, label='Temporal MultiFix', color='red')
            plt.fill_between(x, y_train_tmf - y_train_tmf_std, y_train_tmf + y_train_tmf_std, color='red', alpha=0.3, label='_nolegend_')
            plt.plot(x, y_train_tmf_dw, label='Temporal MultiFix DW', color='blue')
            plt.fill_between(x, y_train_tmf_dw - y_train_tmf_dw_std, y_train_tmf_dw + y_train_tmf_dw_std, color='blue', alpha=0.3, label='_nolegend_')
            plt.plot(x, y_train_mlegcn, label='MultiLabel Evolve GCN', color='green')
            plt.fill_between(x, y_train_mlegcn - y_train_mlegcn_std, y_train_mlegcn + y_train_mlegcn_std, color='green', alpha=0.3, label='_nolegend_')

            plt.xlabel("Epoch")
            plt.ylabel(f"{title} (mean ± std)")
            plt.title(f'alpha={param['ALPHA']}, inter homophily={inter_homophily:.2f}')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            filename = f"{param['IMAGE_FILE']}_{imgname}-1.png"
            filepath = os.path.join(param['EXPERIMENT_PATH'], filename)
            plt.savefig(filepath)
            plt.close()

            # plot 3
            plt.figure(figsize=(8, 5))
            plt.errorbar(x, y_train_tmf, yerr=y_train_tmf_std, fmt='o-', capsize=5, label='Temporal MultiFix', color='red')
            plt.errorbar(x, y_train_tmf_dw, yerr=y_train_tmf_dw_std, fmt='o-', capsize=5, label='Temporal MultiFix DW',
                        color='blue')
            plt.errorbar(x, y_train_mlegcn, yerr=y_train_mlegcn_std, fmt='o-', capsize=5, label='MultiLabel Evolve GCN',
                        color='green')

            plt.xlabel("Epoch")
            plt.ylabel(f"{title} (mean ± std)")
            plt.title(f'alpha={param['ALPHA']}, inter homophily={inter_homophily:.2f}')
            plt.grid(True)
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            filename = f"{param['IMAGE_FILE']}_{imgname}-zoom.png"
            filepath = os.path.join(param['EXPERIMENT_PATH'], filename)
            plt.savefig(filepath)
            plt.close()

            # plot 4
            plt.figure(figsize=(8, 5))
            plt.plot(x, y_train_tmf, label='Temporal MultiFix', color='red')
            plt.fill_between(x, y_train_tmf - y_train_tmf_std, y_train_tmf + y_train_tmf_std, color='red', alpha=0.3, label='_nolegend_')
            plt.plot(x, y_train_tmf_dw, label='Temporal MultiFix DW', color='blue')
            plt.fill_between(x, y_train_tmf_dw - y_train_tmf_dw_std, y_train_tmf_dw + y_train_tmf_dw_std, color='blue', alpha=0.3, label='_nolegend_')
            plt.plot(x, y_train_mlegcn, label='MultiLabel Evolve GCN', color='green')
            plt.fill_between(x, y_train_mlegcn - y_train_mlegcn_std, y_train_mlegcn + y_train_mlegcn_std, color='green', alpha=0.3, label='_nolegend_')

            plt.xlabel("Epoch")
            plt.ylabel(f"{title} (mean ± std)")
            plt.title(f'alpha={param['ALPHA']}, inter homophily={inter_homophily:.2f}')
            plt.grid(True)
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            filename = f"{param['IMAGE_FILE']}_{imgname}-zoom-1.png"
            filepath = os.path.join(param['EXPERIMENT_PATH'], filename)
            plt.savefig(filepath)
            plt.close()



def experiment_main(param):
    # setup export, establish folder to store all results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    param['EXPERIMENT_PATH'] = os.path.join(param['EXPERIMENT_PATH'], timestamp)
    os.makedirs(param['EXPERIMENT_PATH'], exist_ok=True)

    config = TemporalMultiLabelGeneratorConfig(m_rel=param['NUM_REL_FEATURES'],
                                                m_irr=param['NUM_IRR_FEATURES'],
                                                m_red=param['NUM_RED_FEATURES'],
                                                q=param['NUM_LABELS'],
                                                N=param['NUM_NODES'],
                                                max_r=0.8,
                                                min_r=((param['NUM_LABELS'] / 10) + 1) / param['NUM_LABELS'],
                                                mu=0,
                                                b=0.05,
                                                alpha=param['ALPHA'],
                                                theta=np.pi / -(param['NUM_TIMESTEPS']//-2), #ceiling division
                                                horizon=param['NUM_TIMESTEPS'],
                                                sphere_sampling='polar',
                                                data_sampling='global',
                                                rotation_reference='data',
                                                )
    
    dataset_loader = DatasetLoader(config, param['EMBEDDING_DIM'], param['FILENAME'], param['BASEFILE'])
    
    # alphas = [5, 4, 3, 2, 1, 0]
    # alphas = [6, 7, 8, 9, 10]
    alphas = [0.5, 1.5, 2.5]
    for alpha in alphas:
        # experiment 
        param['ALPHA'] = alpha
        dataset_loader.generator.alpha = alpha
        train_dataset, test_dataset, train_embedding, test_embedding, inter_homophily, intra_homophily = load_data(dataset_loader, param)
        
        datasets = {'train_data' : train_dataset,
                    'test_data' : test_dataset,
                    'train_emb' : train_embedding,
                    'test_emb' : test_embedding,
                    'inter_homophily' : inter_homophily,
                    'intra_homophily' : intra_homophily,
                    }

        experiment_repeats(param, datasets)
    # experiment_single_run(param)



if __name__ == "__main__":
    print(f'{DEVICE} available')
    experiment_main(PARAM)

