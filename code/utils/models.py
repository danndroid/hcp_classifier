import os.path as osp
import numpy as np
import pandas as pd
from scipy.stats import zscore

import torch
import torch.nn as nn
import torch_geometric

from torch_geometric.nn import GCNConv#, SAGEConv, GATConv, GATv2Conv, GCNConv, GlobalAttention
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import global_add_pool

class Attention2Conv(nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout_rate=0.5, edge_dropout_rate=0.1):
        super(Attention2Conv, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.relu1 = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)#, dropout=dropout)
        self.relu2 = nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        self.edge_drop = edge_dropout_rate
        
        # Attention-based graph-level aggregation
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_channels, 1))
        nn.init.xavier_uniform_(self.attention_weights)  # Initialize attention weight

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):

        edge_index, _ = dropout_edge(edge_index, p=self.edge_drop, training=self.training)

        node_embeddings = self.conv1(x, edge_index)
        node_embeddings = self.bn1(node_embeddings)
        node_embeddings = self.relu1(node_embeddings)
        node_embeddings = self.dropout1(node_embeddings)

        node_embeddings = self.conv2(node_embeddings, edge_index)
        node_embeddings = self.bn2(node_embeddings)
        node_embeddings = self.relu2(node_embeddings)
        node_embeddings = self.dropout2(node_embeddings)

        # Step 2: Compute attention scores for each node in the graph
        attention_scores = torch.matmul(node_embeddings, self.attention_weights)  # (num_nodes, 1)
        attention_scores = torch.sigmoid(attention_scores)  # Normalize scores to [0,1]
        
        # Step 3: Compute graph embedding as weighted sum of node embeddings
        weighted_embeddings = node_embeddings * attention_scores  # (num_nodes, hidden_dim * num_heads)
        graph_embedding = global_add_pool(weighted_embeddings, batch)  # (num_graphs, hidden_dim * num_heads)

        # Step 4: Compute logits for binary classification
        logits = self.classifier(graph_embedding)  # (num_graphs, 1)
        
        return logits, attention_scores  # Return logits and node attention scores





class Attention3Conv(nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout_rate=0.5, edge_dropout_rate=0.1):
        super(Attention3Conv, self).__init__()

        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.relu1 = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)#, dropout=dropout)
        self.relu2 = nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)#, dropout=dropout)
        self.relu3 = nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)
        self.dropout3 = torch.nn.Dropout(dropout_rate)

        self.edge_drop = edge_dropout_rate
        
        
        # Attention-based graph-level aggregation
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_channels, 1))
        nn.init.xavier_uniform_(self.attention_weights)  # Initialize attention weight

        # Final classifier
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):

        edge_index, _ = dropout_edge(edge_index, p=self.edge_drop, training=self.training)

        node_embeddings = self.conv1(x, edge_index)
        node_embeddings = self.bn1(node_embeddings)
        node_embeddings = self.relu1(node_embeddings)
        node_embeddings = self.dropout1(node_embeddings)

        node_embeddings = self.conv2(node_embeddings, edge_index)
        node_embeddings = self.bn2(node_embeddings)
        node_embeddings = self.relu2(node_embeddings)
        node_embeddings = self.dropout2(node_embeddings)

        node_embeddings = self.conv3(node_embeddings, edge_index)
        node_embeddings = self.bn3(node_embeddings)
        node_embeddings = self.relu3(node_embeddings)
        node_embeddings = self.dropout3(node_embeddings)

        # Step 2: Compute attention scores for each node in the graph
        attention_scores = torch.matmul(node_embeddings, self.attention_weights)  # (num_nodes, 1)
        attention_scores = torch.sigmoid(attention_scores)  # Normalize scores to [0,1]
        
        # Step 3: Compute graph embedding as weighted sum of node embeddings
        weighted_embeddings = node_embeddings * attention_scores  # (num_nodes, hidden_dim * num_heads)
        graph_embedding = global_add_pool(weighted_embeddings, batch)  # (num_graphs, hidden_dim * num_heads)

        # Step 4: Compute logits for binary classification
        logits = self.classifier(graph_embedding)  # (num_graphs, 1)
        
        return logits, attention_scores  # Return logits and node attention scores




    

    
     
