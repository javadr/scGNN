import numpy as np
import scipy.io as io
from scipy import sparse
import pandas as pd
from config import CFG
from pathlib import Path
from typing import Optional, Callable, Union, Tuple

import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
import networkx as nx
import copy

# system inits
torch.manual_seed(CFG.seed)
torch.cuda.manual_seed_all(CFG.seed)
np.random.seed(CFG.seed)


# Data Preprocessing
class DataSanitizer():

    def __init__(self, fn: Optional[str] = None, data_path:  Optional[str] = None) -> None:
        self.file_name = fn if fn else 'matrix.mtx'
        # `matrix.mtx` is a sparse matrix in COOrdinate format.
        # num_cells x num_genes
        path = Path(data_path) if data_path else CFG.base_path/"data"
        self.data = io.mmread(path/self.file_name).T
        # Original data in a matrix form with dimension of (cells x genes)
        self.raw = pd.DataFrame(self.data.todense())
        self.data_norm = self.normalize()
        # number of nonzero items
        self.masked_prob = min(self.data.nnz / (self.data.shape[0] * self.data.shape[1]), 0.3)

    @staticmethod
    def filter_genes(data:pd.DataFrame, threshold:float, method:str='var2mean+1') -> pd.Index:
        """
            data: pandas.DataFrame with rows as cells and columns as genes
            return: indices of selected genes
        """
        if method=='var2mean+1':
            genes = (data.var()/(1+data.mean())).sort_values(ascending=False)
            genes = genes[genes > 0]
            lim = (genes > threshold).sum()
            genes_to_impute = genes.index[:lim]
            print(f"{len(genes_to_impute)} genes selected for imputation")
        else:
            raise NameError("choosen method does not exist!")
        return genes_to_impute

    def normalize(self) -> np.ndarray:
        """ Normalize data  (as pandas dataframe)"""
        genes_to_impute = self.filter_genes(self.raw, CFG.gene_threshold)
        self.data = self.data.tocsr()[:, genes_to_impute]
        # normalize cell counts
        rowsum = self.data.sum(axis = 1)
        norm_transcript = np.median(np.asarray(rowsum))
        data_norm = (self.data/(rowsum+1))*norm_transcript # shape -> (#cells, #selected genes)
        return np.asarray(data_norm) # numpy.matrix will be deprecated soon

    def get(self) -> np.ndarray:
        """ Selected Normalized data in matrix form (as pandas dataframe)"""
        return self.data_norm

    def get_raw(self) -> pd.DataFrame:
        """ Original data in matrix form (as pandas dataframe)"""
        return self.raw

class CellGraph(Dataset):
    def __init__(self,data: np.ndarray, device, transform: Optional[Callable]=None, pre_transform: Optional[Callable]=None ):
        super().__init__(None, transform, pre_transform)
        self.data = copy.deepcopy(data)
        x_size = data.shape[0] # number of cells
        train_mask, val_mask, test_mask = CellGraph.split_mask(x_size)
        index_mask, index_zeros = self.mask(masked_prob=CFG.masked_prob) # mask some of non-zero data

        edges = CellGraph.build_cell_graph(self.data)
        x_features = torch.tensor(self.data, dtype=torch.float)
        self.Graph = Data(x=x_features, edge_index=edges, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask).to(device=device)
        self.Graph.index_mask = index_mask # rows/columns of masked non-zero elements
        self.Graph.index_zeros = index_zeros # rows/columns of all zero elements

    def __call__(self,) -> Data:
        return self.Graph

    def len(self, ):
        return 1

    def get(self, idx: int) -> Data:
        r"""Gets the data object at index :obj:`idx`."""
        return self.Graph

    def mask(self, masked_prob:float) -> Tuple[torch.Tensor, torch.Tensor]:
        idx_nonzero = np.where(self.data != 0)
        size = idx_nonzero[0].size
        masking_idx = np.random.choice(size, int(size*masked_prob), replace = False)
        self.data[idx_nonzero[0][masking_idx], idx_nonzero[1][masking_idx]] = 0
        idx_zeros   = np.where(self.data == 0)

        # print(f"Non-zero items: {size}\nmask: {masking_idx.shape}\nOriginal Data: {self.data.shape}")
        index_mask = torch.zeros(self.data.shape, dtype=torch.bool, requires_grad=False)
        index_zeros = index_mask.data.clone()
        index_mask[idx_nonzero[0][masking_idx], idx_nonzero[1][masking_idx]] = True
        index_zeros[idx_zeros[0], idx_zeros[1]] = True
        return index_mask, index_zeros


    @staticmethod
    def split_mask(x_size: int, ratio_train: float=CFG.ratio_train, ratio_val_to_test: float=CFG.ratio_val_to_test):
        perm = np.random.permutation(x_size)
        train_size = int(x_size * ratio_train)
        ratio_val = ratio_val_to_test / (1 + ratio_val_to_test)
        val_size = int(ratio_val * (x_size - train_size))

        train_mask = torch.zeros(x_size, dtype=torch.bool, requires_grad=False)
        train_mask[torch.tensor(perm[:train_size])] = True

        val_mask = torch.zeros(x_size, dtype=torch.bool, requires_grad=False)
        val_mask[torch.tensor(perm[train_size:train_size+val_size])] = True

        test_mask = torch.zeros(x_size, dtype=torch.bool, requires_grad=False)
        test_mask[torch.tensor(perm[train_size+val_size:])] = True

        return train_mask, val_mask, test_mask


    @staticmethod
    def build_cell_graph(X:np.ndarray, method='pca_knn_graph'):
        data_pca = CellGraph.PCA(X)
        if method == 'pca_knn_graph':
            edge_index = CellGraph.edge_index(data_pca)
        elif method == 'pca_kneighbors_graph':
            edge_index = CellGraph.edge_index(data_pca, method='kneighbors_graph')
        else:
            raise NameError('Unknown method')
        return edge_index


    @staticmethod
    def PCA(X:np.ndarray, n_components=CFG.n_components):
        # apply pca to construct graph
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(X)
        print(f"Reduced data shape is {data_pca.shape}.")
        print(pca.explained_variance_ratio_[:10])
        return data_pca

    @staticmethod
    def edge_index(data: np.ndarray, k=CFG.n_neighbors, method='knn_graph'):
        assert method in ['knn_graph', 'kneighbors_graph']
        if method == 'knn_graph':
            edges = knn_graph(torch.tensor(data), k=k)
        elif method == 'kneighbors_graph':
            A = kneighbors_graph(data, k, mode='connectivity', include_self=False)
            G = nx.from_numpy_matrix(A.todense())
            # prepare for pytorch geometric data loading
            edges = np.array( list(G.edges()) + [(v,u) for u,v in G.edges()] ).T
            edges = torch.tensor(edges, dtype = torch.long)
        else:
            raise NameError('Unknown method')
        return edges