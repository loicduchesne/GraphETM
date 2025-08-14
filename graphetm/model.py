### Imports
# Local
from .encoder import Encoder
from .decoder import Decoder
from .gcn import GCN

# External
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphETM(nn.Module):
    """
    GraphETM model with an Embedded Topic Model (ETM) Encoder and Decoder and a Graph Convolutional Network (GCN) Filter.
    """
    def __init__(
            self,
            encoder_params: Dict[str, Dict[str, int]],
            gcn_params: Dict[str, int],
            graph_recon_loss: torch.nn.Module,
            # Embeddings:
            embedding: torch.Tensor,
            edge_index: torch.Tensor,
            id_embed_sc: np.ndarray,
            id_embed_ehr: np.ndarray,
            # Params:
            theta_act: str = 'relu',
            num_topics: int = 25,
            dropout: float = 0.2,
            device: torch.device = torch.device('cpu'),

    ):
        """
        Args:
            encoder_params (dict): Dictionary of the parameters for the encoders. Dictionary {'sc': {str: Any},
                'ehr': {str: Any}}. (vocab_size: Size of vocabulary, encoder_hidden_size: Size of the hidden layer in
                the encoder).
            gcn_params: (dict): Dictionary for the GCN parameters.
            graph_recon_loss (torch.nn.Module): Graph reconstruction loss function for the GCN.
            embedding (torch.Tensor): Initial embedding (also known as rho) computed from the knowledge graph (e.g.:
                TransE embeddings).
            edge_index (torch.Tensor): The edge indices for the embedding matrix.
            id_embed_sc (np.ndarray): Array of indices corresponding to genes in the embedding matrix from the scRNA BoW
                data.
            id_embed_ehr (np.ndarray): Array of indices corresponding to diseases in the embedding matrix from the EHR BoW
                data.
            theta_act (str): Activation function for theta.
                Default: 'relu'.
            num_topics (int): Number of topics.
                Default: 25.
            dropout (float): Dropout rate.
                Default: 0.2.
            device (torch.device): Device to be used.
                Default: torch.device('cpu').
        """

        super(GraphETM, self).__init__()

        self.encoder_params = encoder_params
        self.graph_recon_loss = graph_recon_loss.to(device)

        ## Embeddings
        embedding_dim = embedding.shape[1]
        self.register_buffer('embedding', embedding) # [N_total, L]
        self.edge_index = edge_index.to(device)
        self.id_embed_sc  = torch.tensor(id_embed_sc , dtype=torch.long, device=device)
        self.id_embed_ehr = torch.tensor(id_embed_ehr, dtype=torch.long, device=device)

        ## Layers
        self.gcn = GCN(**gcn_params, embedding_dim=embedding_dim)
        self.enc_sc  = Encoder(**encoder_params['sc'],  num_topics=num_topics, dropout=dropout, theta_act=theta_act)
        self.enc_ehr = Encoder(**encoder_params['ehr'], num_topics=num_topics, dropout=dropout, theta_act=theta_act)
        self.dec_sc  = Decoder(embedding_dim=embedding_dim, num_topics=num_topics)
        self.dec_ehr = Decoder(embedding_dim=embedding_dim, num_topics=num_topics)

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """
            Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def infer_topic_distribution(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """
            Returns a deterministic topic distribution for evaluation purposes bypassing the stochastic reparameterization step.

            Args:
                normalized_bows (torch.Tensor): Normalized bag-of-words input.

            Returns:
                torch.Tensor: Deterministic topic proportions.
        """
        theta = self.encoder.infer_topic_distribution(normalized_bows) # FIXME: To fix.
        return theta

    def step_forward(self, encoder, decoder, bow, rho, aggregate=True):
        bow_raw  = bow # integer counts
        lengths  = bow_raw.sum(1, keepdim=True).clamp(min=1e-8)
        bow_norm = bow_raw / lengths # Normalize

        mu, logvar, kld = encoder(bow_norm)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1) # [D, K] (batch_size, num_topics)

        preds = decoder(theta, rho=rho)
        rec_loss = -(preds * bow_raw).sum(1)
        if aggregate:
            rec_loss = rec_loss.mean() / lengths.mean()
        else:
            rec_loss = rec_loss / lengths.squeeze(1)

        return ({'theta': theta.detach(), 'preds': preds.detach()}, # Outputs
                {'rec_loss': rec_loss,'kld': kld})                   # Losses

    def forward(self, bow_sc, bow_ehr):
        """
        Forward pass for the multi-modal Embedded Topic Model (ETM).
        Args:
            bow_sc : Bag-of-words representation of single-cell (SC) RNA modality.
            bow_ehr: Bag-of-words representation of Electronic Health Record (EHR) modality.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Tuple (outputs, losses) with outputs and losses
            dictionaries. Access the modality outputs with keys 'sc' or 'ehr' for either dictionary. Modalities within
            outputs contain keys 'theta' and 'preds'. Modalities within losses contain keys 'rec_loss' and 'kld'.

        """
        # Filter: Forward
        rho_full_new = self.gcn.forward(embedding=self.embedding, edge_index=self.edge_index) # [N_total, L]
        graph_loss = self.graph_recon_loss(rho_full_new, edge_index=self.edge_index)

        rho_sc  = rho_full_new[self.id_embed_sc]  # [V_sc , L]
        rho_ehr = rho_full_new[self.id_embed_ehr] # [V_ehr, L]

        # Encoder-Decoder: ScRNA
        outputs_sc, losses_sc = self.step_forward(
            bow=bow_sc,
            encoder=self.enc_sc,
            decoder=self.dec_sc,
            rho=rho_sc)

        # Encoder-Decoder: EHR
        outputs_ehr, losses_ehr = self.step_forward(
            bow=bow_ehr,
            encoder=self.enc_ehr,
            decoder=self.dec_ehr,
            rho=rho_ehr)


        return ({'sc': outputs_sc, 'ehr': outputs_ehr}, # Outputs
                {'sc': losses_sc , 'ehr': losses_ehr, 'graph_loss': graph_loss})  # Losses