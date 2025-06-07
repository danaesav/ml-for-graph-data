import torch as th
import numpy as np
from torch_geometric.nn import Node2Vec
from gensim.models.word2vec import Word2Vec
from typing import Literal, List

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = 'cuda' if th.cuda.is_available() else 'cpu'

class NodeEmbedding:

    def __init__(self, embedding_dim=128, walk_length=10, window_size=5, walks_per_node=5):

        self.embedding_dim = embedding_dim # (2, num_edges*2) *2 for bidirection
        self.walk_length = walk_length
        self.window_size = window_size
        self.walks_per_node = walks_per_node


    def get_embedding(self, edge_index, adj_mat, method:Literal['Node2Vec', 'DeepWalk']):

        if method == 'Node2Vec':
            return self._node2vec(edge_index)
        
        else:
            # this implementation is no good
            return self._deepwalk(adj_mat)
        

    def _node2vec(self, edge_index, epoch=50):
        self.model = Node2Vec(th.tensor(edge_index, dtype=th.long),
                              embedding_dim=self.embedding_dim,
                              walk_length=self.walk_length,
                              context_size=self.window_size,
                              walks_per_node=self.walks_per_node,
                              sparse=True,
                              ).to(DEVICE)
        

        self.loader = self.model.loader(batch_size=128, shuffle=True)
        self.optimizer = th.optim.SparseAdam(list(self.model.parameters()), lr=0.01)

        return self._n2v_embeddings(epoch)
    
    def _n2v_train(self):

        self.model.train()
        total_loss = 0

        for pos_rw, neg_rw in self.loader:
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()


        return total_loss / len(self.loader)
    

    def _n2v_embeddings(self, epoch):

        for i in range(epoch):
            loss = self._n2v_train()

        with th.no_grad():
            self.model.eval()
            z = self.model()  # This returns the embeddings
            return z.cpu()
        


    def _deepwalk(self, adj_mat):
        
        walks = self._dw_get_walks(adj_mat)
        return self._dw_compute_embeddings(walks)


    def _dw_random_walk(self, adj_mat, start: int):
        """
        Generate a random walk starting on start
        """
        walk = [start]

        for i in range(self.walk_length):
            neighbours = np.where(adj_mat[walk[i]])[0]

            if neighbours.size == 0:
                break

            p = np.random.choice(neighbours)
            walk.append(p)

        return walk
    
    def _dw_get_walks(self, adj_mat) -> List[List[int]]:
        """
        Generate all the random walks
        """
        random_walks = []
        for _ in range(self.walks_per_node):
            random_nodes = np.arange(adj_mat.shape[0])
            np.random.shuffle(random_nodes)
            for node in random_nodes:
                random_walks.append(self._dw_random_walk(adj_mat, start=node))
        return random_walks
    

    def _dw_compute_embeddings(self, walks: List[List[int]]):
        """
        Compute the node embeddings for the generated walks
        """
        model = Word2Vec(sentences=walks, window=self.window_size, vector_size=self.embedding_dim)
        return model.wv
