import torch
import torch.nn.functional as F

#from torch_geometric.data import Data
#from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, MessagePassing, global_mean_pool
from torch_geometric.nn import Set2Set

from torch_geometric.nn import GlobalAttention


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        # First GCN layer
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)  # BatchNorm for first layer
        self.conv2 = GATConv(hidden_dim, hidden_dim//2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim//2)  # BatchNorm for second layer
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim//2, output_dim)  # Fully connected layer



    def forward(self, data):
        # Extract data components
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Global mean pooling to aggregate node-level features into graph-level features
        x = global_mean_pool(x, batch)

        # Fully connected layer for graph-level classification/regression
        out = self.fc(x)
        return out
