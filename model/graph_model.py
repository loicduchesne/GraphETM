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
            edge_index: torch.Tensor
    ):
        super(GraphModel, self).__init__()

        self.register_buffer('rho_full0', embedding)  # [N_total, L]
        self.edge_index = edge_index

        self.conv1 = GCNConv(embedding_dim, embedding_dim) # [L, ]
        #self.conv2 = GCNConv(32, embedding_dim) # [, L]

    def forward(self): # TODO: Normalize/dropout potentially to be added.
        rho_full1 = self.conv1(self.rho_full0, self.edge_index) # [N_total, L]
        rho_full1 = F.relu(rho_full1)
        return rho_full1




## LINEAR FILTER BLOCK
class old_Filter(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            embedding: torch.Tensor
    ):
        super(old_Filter, self).__init__()

        self.register_buffer('rho_full0', embedding)  # [N_total, L]

        self.conv1 = nn.Linear(embedding_dim, embedding_dim) # [L, L]

    def forward(self): # TODO: Normalize/dropout potentially to be added.
        rho_full1 = self.conv1(self.rho_full0) # [N_total, L]
        rho_full1 = F.relu(rho_full1)
        return rho_full1