import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

### ELBO
class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()


### GRAPH RECONSTRUCTION LOSS
# FULL LOSS
def _full_graph_recon_loss(h: torch.Tensor, edge_index: torch.LongTensor):
    # Positive edges
    pos_edge_index = edge_index  # shape [2, E_pos]

    # Negative edges
    neg_edge_index = pyg_utils.negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=h.size(0),
        num_neg_samples=pos_edge_index.size(1))

    # Gather embeddings
    src_pos = h[pos_edge_index[0]] # [E_pos, out_dim]
    dst_pos = h[pos_edge_index[1]] # [E_pos, out_dim]
    src_neg = h[neg_edge_index[0]] # [E_pos, out_dim]
    dst_neg = h[neg_edge_index[1]] # [E_pos, out_dim]

    # Inner-product score
    pos_scores = (src_pos * dst_pos).sum(dim=1)
    neg_scores = (src_neg * dst_neg).sum(dim=1)

    # Compute loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return pos_loss + neg_loss

# SUBSAMPLED LOSS
def _subsampled_graph_recon_loss(h: torch.Tensor, edge_index: torch.LongTensor, subsampling_size: int):
    # Positive edges
    pos_edge_index = edge_index  # shape [2, E_pos]
    E_pos = pos_edge_index.size(1)

    perm = torch.randperm(E_pos, device=pos_edge_index.device)[:subsampling_size]
    pos_edge_index_sub = pos_edge_index[:, perm] # shape [2, subsampling_size]

    # Negative edges
    neg_edge_index_sub = pyg_utils.negative_sampling(
        edge_index=pos_edge_index_sub,
        num_nodes=h.size(0),
        num_neg_samples=subsampling_size)

    # Gather embeddings
    src_pos = h[pos_edge_index_sub[0]]
    dst_pos = h[pos_edge_index_sub[1]]
    src_neg = h[neg_edge_index_sub[0]] # Dev. note: This may have been the source of the issue for graph rec. loss.
    dst_neg = h[neg_edge_index_sub[1]]

    # Inner-product score
    pos_scores = (src_pos * dst_pos).sum(dim=1)
    neg_scores = (src_neg * dst_neg).sum(dim=1)

    # Compute loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    return pos_loss + neg_loss

# MAIN FUNCTION
def graph_recon_loss(h: torch.Tensor, edge_index: torch.LongTensor, subsampling_size: int = None):
    if subsampling_size is not None:
        loss = _subsampled_graph_recon_loss(h, edge_index, subsampling_size)
    else:
        loss = _full_graph_recon_loss(h, edge_index)
    return loss