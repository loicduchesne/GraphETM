### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int = 64,
    ):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(embedding_dim, hidden_dim) # [L, ]
        self.conv2 = GCNConv(hidden_dim, embedding_dim) # [, L]

    def forward(self, embedding: torch.Tensor, edge_index: torch.LongTensor):
        rho_full1 = F.relu(self.conv1(embedding, edge_index)) # [N_total, L]
        rho_full2 = F.relu(self.conv2(rho_full1, edge_index))
        return rho_full2