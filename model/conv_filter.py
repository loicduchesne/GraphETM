### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


### GRAPH-ETM MODEL ARCHITECTURE
## GRAPH-CONVOLUTIONAL FILTER BLOCK
class GraphFilter(nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            embedding_dim: int,
            edge_index: torch.LongTensor,
    ):
        super(GraphFilter, self).__init__()

        self.edge_index = edge_index

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index) # [N, embedding_dim]
        # TODO: Normalize/dropout potentially to be added.
        return h