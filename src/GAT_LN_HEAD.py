import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT_LN_HEAD(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, num_heads=4):
        super(GAT_LN_HEAD, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)  # Corrected input_dim
        self.ln1 = torch.nn.LayerNorm(normalized_shape=hidden_dim * num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim // 2, heads=num_heads)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=(hidden_dim // 2) * num_heads)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.fc = torch.nn.Linear((hidden_dim // 2) * num_heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        return self.fc(x)
