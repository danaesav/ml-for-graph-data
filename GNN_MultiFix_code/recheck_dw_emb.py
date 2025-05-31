import pandas as pd
import numpy as np
import torch
import os.path as osp
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data
from gensim.models import Word2Vec, KeyedVectors

# class TopKRanker(OneVsRestClassifier):
#     def predict(self, X, top_k_list):
#         assert X.shape[0] == len(top_k_list)
#         probs = np.asarray(super(TopKRanker, self).predict_proba(X))
#         all_labels = []
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-k:]].tolist()
#             all_labels.append(labels)
#         return all_labels

def load_hyper_data(data_name="Hyperspheres_10_10_0", split_name="split_0.pt", train_percent=0.6, path="data/"):

    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(osp.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()


    # load split
    folder_name = data_name + "_" + str(train_percent)
    file_path = osp.join(path, folder_name, split_name)
    masks = torch.load(file_path)
    train_idx = masks["train_mask"]
    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_mask[train_idx] = True

    val_idx = masks["val_mask"]
    val_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    val_mask[val_idx] = True

    test_idx = masks["test_mask"]
    test_mask = torch.zeros(labels.shape[0], dtype=torch.bool)
    test_mask[test_idx] = True

    G = Data(y=labels)

    G.train_mask = train_mask
    G.val_mask = val_mask
    G.test_mask = test_mask
    G.num_nodes = G.y.shape[0]

    # load deep walk embeddings
    emb_model = KeyedVectors.load_word2vec_format("data/Hyperspheres_10_10_0/hypersphere.emb", binary=False)
    deep_walk_emb = np.asarray([emb_model[str(node)] for node in range(G.num_nodes)])
    G.deep_walk_emb = torch.from_numpy(deep_walk_emb).float()


    return G


G = load_hyper_data(split_name="split_0.pt", train_percent=0.6)


# Load your embeddings and labels data
# Assuming embeddings is a numpy array and labels is a list of lists
embeddings = G.deep_walk_emb
labels = G.y

# top_k_list = [torch.nonzero(l).shape[0] for l in labels]

# # Split your data into training and testing sets
X_train, X_test, y_train, y_test = embeddings[G.train_mask], embeddings[G.test_mask], labels[G.train_mask],  labels[G.test_mask]

# # Initialize a logistic regression model with multi-label support
# model = TopKRanker(LogisticRegression())
model = OneVsRestClassifier(LogisticRegression())
# Train the model on your training data
model.fit(X_train, y_train)

# Make predictions on your test data
# mask = G.test_mask
# indices = torch.where(mask)[0]
# top_k_list_indices = [top_k_list[i] for i in indices.tolist()]

#y_pred = model.predict(X_test, top_k_list_indices)
y_pred = model.predict(X_test)
# # one hot encode
# from sklearn.preprocessing import MultiLabelBinarizer
# # Assuming y_pred is your list of lists of predictions
# mlb = MultiLabelBinarizer(classes=np.arange(20))
# y_pred_onehot = mlb.fit_transform(y_pred)


# Evaluate the model using average precision score
#average_precision = average_precision_score(y_test, y_pred_onehot)
average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))