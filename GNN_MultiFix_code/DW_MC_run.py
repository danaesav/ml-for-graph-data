from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import Planetoid
from data_loader import load_cora, load_citeseer
# Load the Cora and Citeseer datasets
cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
citeseer_dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

# load the embeddings
G_cora = load_cora()
G_citeseer = load_citeseer()

cora_embeddings = G_cora.deep_walk_emb
citeseer_embeddings = G_citeseer.deep_walk_emb


# Train a logistic regression model for the Cora dataset
cora_model = LogisticRegression()
cora_model.fit(cora_embeddings[cora_dataset[0].train_mask], cora_dataset[0].y[cora_dataset[0].train_mask])

# Evaluate the model on the test data
cora_pred = cora_model.predict(cora_embeddings[cora_dataset[0].test_mask])
cora_acc = accuracy_score(cora_dataset[0].y[cora_dataset[0].test_mask], cora_pred)
print(f'Cora accuracy: {cora_acc}')

# Train a logistic regression model for the Citeseer dataset
citeseer_model = LogisticRegression()
citeseer_model.fit(citeseer_embeddings[citeseer_dataset[0].train_mask], citeseer_dataset[0].y[citeseer_dataset[0].train_mask])

# Evaluate the model on the test data
citeseer_pred = citeseer_model.predict(citeseer_embeddings[citeseer_dataset[0].test_mask])
citeseer_acc = accuracy_score(citeseer_dataset[0].y[citeseer_dataset[0].test_mask], citeseer_pred)
print(f'Citeseer accuracy: {citeseer_acc}')