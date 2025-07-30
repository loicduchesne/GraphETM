### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


### GRAPH-ETM MODEL ARCHITECTURE
## GRAPH-CONVOLUTIONAL FILTER BLOCK
class GraphModel(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            embedding: torch.Tensor,
            edge_index: torch.Tensor,
            hidden_dim: int = 64,
    ):
        super(GraphModel, self).__init__()

        self.register_buffer('rho_full0', embedding)  # [N_total, L]
        self.edge_index = edge_index

        self.conv1 = GCNConv(embedding_dim, hidden_dim) # [L, ]
        self.conv2 = GCNConv(hidden_dim, embedding_dim) # [, L]

    def forward(self): # TODO: Normalize/dropout potentially to be added.
        rho_full1 = F.relu(self.conv1(self.rho_full0, self.edge_index)) # [N_total, L]
        rho_full2 = F.relu(self.conv2(rho_full1, self.edge_index))
        return rho_full2