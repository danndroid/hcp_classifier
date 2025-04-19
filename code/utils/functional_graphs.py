import os.path as osp
import numpy as np
import pandas as pd
from scipy.stats import zscore

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset

import networkx as nx


def compute_graph_measures(fc):
    G = nx.from_numpy_array(fc)
    
    in_centrality = pd.Series(nx.eigenvector_centrality_numpy(G, max_iter=1000)).sort_index().values
    #out_centrality = pd.Series(nx.eigenvector_centrality_numpy(G.reverse(), max_iter=1000)).sort_index().values
    clustering = pd.Series(nx.clustering(G)).sort_index().values
    #degree = np.array([G.degree(node) for node in G.nodes()])

    measures = np.vstack([in_centrality, clustering]).T

    return measures
    

def compute_degree_features(fc):
    
    np.fill_diagonal(fc, 0)
    pfc = np.where(fc>0,fc,0)
    nfc = np.where(fc<0,fc,0)
    
    strength = np.where(fc!=0,fc,0).sum(axis=1)
    degree = np.where(fc!=0,1,0).sum(axis=1)
    p_strength = np.where(pfc>0,pfc,0).sum(axis=1)
    p_degree = np.where(pfc>0,1,0).sum(axis=1)
    n_strength = np.where(nfc<0,nfc,0).sum(axis=1)
    n_degree = np.where(nfc<0,1,0).sum(axis=1)

    measures = np.vstack([degree, p_degree, n_degree, strength, p_strength, n_strength]).T

    return measures
    

def compute_edge_features(fc, edge_list):
    #G = nx.from_numpy_array(fc)
    #edge_list = np.array(G.edges)
    i_coords, j_coords = zip(*edge_list)
    edge_features = fc[i_coords, j_coords]
    
    return edge_features





def create_graph_dataset(sparse_fcs, root, labels, scale_features=False):
    
    dataset_node_features = []
    dataset_edge_list = []
    dataset_edge_features = []
    dataset_labels = []

    N = sparse_fcs.shape[0]
    
    for idx, fc in zip(np.arange(N), sparse_fcs):
        #print(idx)

        G = nx.from_numpy_array(fc)
        edge_list = np.array(G.edges)

        try:
            graph_measures = compute_graph_measures(fc)
        except:
            print(f'Graph {idx}')
            graph_measures = np.zeros((116,3))

        edge_features = compute_edge_features(fc, edge_list)
        degree_features = compute_degree_features(fc)
        #print(degree_features.shape, graph_measures.shape)
        node_features = np.hstack([degree_features, graph_measures])
        
        if scale_features:
            node_features = zscore(node_features)

            
        label = np.array([labels[idx]], dtype='float64')

        # 
        edge_features = torch.from_numpy(edge_features)

        dataset_edge_list.append(edge_list.T)
        dataset_node_features.append(node_features)
        dataset_edge_features.append(edge_features)
        dataset_labels.append(label)

    
    # Create dataset
    dataset = HCPDataset(
        root=root,
        dataset_node_features=dataset_node_features,
        dataset_edge_list=dataset_edge_list,
        dataset_edge_features=dataset_edge_features,
        dataset_labels=dataset_labels
    )
    
    return dataset



class HCPDataset(Dataset):
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

    

    
     
