import os.path as osp
import numpy as np
import pandas as pd
from scipy.stats import zscore

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset

import networkx as nx

def causal_adjacency_list(causal_fc):
    # Works for both PC and GES algorithms
    
    edge_list = np.empty((2, 0), dtype='int64')
    for i,j in zip(np.where(causal_fc!=0)[0], np.where(causal_fc!=0)[1]):
        if causal_fc[j][i]==1 and causal_fc[i][j]==-1: # i --> j
            edge = np.array([[i],[j]], dtype='int64')
        elif causal_fc[j][i]==-1 and causal_fc[i][j]==-1:  # i --- j
            edge = np.array([[i],[j]], dtype='int64')
            #edge = np.array([[i,j],[j,i]], dtype='int64')
        elif causal_fc[j][i]==1 and causal_fc[i][j]==1:  # i <-> j
            edge = np.array([[i],[j]], dtype='int64')
            #edge = np.array([[i,j],[j,i]], dtype='int64')
        else:
            continue

        edge_list = np.concatenate([edge_list, edge], axis=1)
        
    sort_indices = np.lexsort((edge_list[1], edge_list[0]))
    edge_list = edge_list[:, sort_indices]
    
    return edge_list


def compute_edge_features(edge_list, sparse_fc):
    edge_features = sparse_fc[tuple(edge_list)]
    return edge_features


def compute_node_features(causal_fc, sparse_fc):
    
    # degree features
    degree = np.where(causal_fc!=0,1,0).sum(axis=1)
    in_degree = np.where(causal_fc==1,1,0).sum(axis=1)
    out_degree = np.where(causal_fc==-1,1,0).sum(axis=1)

    # strenght features
    in_mask = np.where(causal_fc==1,True,False)
    out_mask = np.where(causal_fc==-1,True,False)
    in_strength = np.where(in_mask, sparse_fc, 0).sum(axis=1)
    in_net_strength = np.where(in_mask, np.abs(sparse_fc), 0).sum(axis=1)
    out_strength = np.where(out_mask, sparse_fc, 0).sum(axis=1)
    out_net_strength = np.where(out_mask, np.abs(sparse_fc), 0).sum(axis=1)

    node_features = np.vstack([degree, in_degree, out_degree, in_strength, in_net_strength, out_strength, out_net_strength]).T
    
    return node_features


def compute_graph_measures(edge_list):
    edge_list = edge_list.T
    edge_list = [list(e) for e in edge_list]
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    in_centrality = pd.Series(nx.eigenvector_centrality_numpy(G, max_iter=1000)).sort_index().values
    out_centrality = pd.Series(nx.eigenvector_centrality_numpy(G.reverse(), max_iter=1000)).sort_index().values
    clustering = pd.Series(nx.clustering(G.reverse())).sort_index().values

    measures = np.vstack([in_centrality, out_centrality, clustering]).T

    return measures


def create_networx_digraph(edge_list):
    edge_list = edge_list.T
    edge_list = [list(e) for e in edge_list]
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    return G


def create_graph_dataset(causal_fcs, sparse_fcs, root, n_controls, scale_features=False):
    
    dataset_node_features = []
    dataset_edge_list = []
    dataset_edge_features = []
    dataset_labels = []

    N = causal_fcs.shape[0]
    
    for idx, causal_fc, sparse_fc in zip(np.arange(N), causal_fcs, sparse_fcs):
        print(idx)

        edge_list = causal_adjacency_list(causal_fc)

        try:
            graph_measures = compute_graph_measures(edge_list)
        except:
            print(f'Graph {idx}')
            graph_measures = np.zeros((116,3))

        edge_features = compute_edge_features(edge_list, sparse_fc)
        degree_features = compute_node_features(causal_fc, sparse_fc)
        #print(degree_features.shape, graph_measures.shape)
        node_features = np.hstack([degree_features, graph_measures])
        
        if scale_features:
            node_features = zscore(node_features)
        
        if idx < n_controls:
            label = np.zeros((1,))
        else:
            label = np.ones((1,))

        edge_features = torch.from_numpy(edge_features)

        dataset_edge_list.append(edge_list)
        dataset_edge_features.append(edge_features)

        
        dataset_node_features.append(node_features)
        dataset_edge_features.append(edge_features)
        dataset_labels.append(label)

    
    # Create dataset
    dataset = AdniDataset(
        root=root,
        dataset_node_features=dataset_node_features,
        dataset_edge_list=dataset_edge_list,
        dataset_edge_features=dataset_edge_features,
        dataset_labels=dataset_labels
    )
    
    return dataset



class AdniDataset(Dataset):
    def __init__(self, root, dataset_node_features, dataset_edge_list, dataset_edge_features=None, dataset_labels=None, transform=None, pre_transform=None):
        """
        Custom dataset creation from numpy arrays.
        
        Args:
            root: Root directory where the dataset should be saved
            features: List of node feature matrices (numpy arrays)
            edge_indices: List of edge index matrices (numpy arrays)
            edge_attrs: List of edge attribute matrices (numpy arrays), optional
            labels: List of graph labels (numpy arrays), optional
            transform: Transform to be applied to each data object
            pre_transform: Transform to be applied to all data objects before saving
        """
        self.features = dataset_node_features
        self.edge_indices = dataset_edge_list
        self.edge_attrs = dataset_edge_features
        self.labels = dataset_labels
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        # No raw files needed for this example
        return []
    
    @property
    def processed_file_names(self):
        # Generate a list of processed file names
        return [f'data_{idx}.pt' for idx in range(len(self.features))]
    
    def download(self):
        # No download required
        pass
    
    def process(self):
        idx = 0
        for i in range(len(self.features)):
            # Convert node features to tensor
            x = torch.tensor(self.features[i], dtype=torch.float)
            
            # Convert edge indices to tensor
            edge_index = torch.tensor(self.edge_indices[i], dtype=torch.long)
            
            # Create data object
            data = Data(x=x, edge_index=edge_index)
            
            # Add edge attributes if provided
            if self.edge_attrs is not None:
                edge_attr = torch.tensor(self.edge_attrs[i], dtype=torch.float)
                data.edge_attr = edge_attr
            
            # Add labels if provided
            if self.labels is not None:
                y = torch.tensor(self.labels[i], dtype=torch.float)
                data.y = y
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            # Save processed data
            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

    

    
     
