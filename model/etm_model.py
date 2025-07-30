### Imports
# Local
from .encoder import Encoder
from .decoder import Decoder

# External
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


### GRAPH-ETM MODEL ARCHITECTURE
## Main GraphETM model.
# Description: MODEL assembles the GraphETM architecture using the ENCODER and DECODER.

# ------------------------------------------------------------------
# @title ETM Model
class ETMModel(nn.Module):
    def __init__(
            self,
            encoder_params  : Dict[str, Dict[str, int]],
            theta_act: str,
            num_topics: int,
            embedding_dim : int,
            dropout = 0.2
    ):
        """
            Initialize the ETM model.

            Args:
                encoder_params: Dictionary of the parameters for the encoders. Dictionary {'sc': {str: Any}, 'ehr': {str: Any}}.
                    vocab_size: Size of vocabulary.
                    encoder_hidden_size: Size of the hidden layer in the encoder.
                theta_act: Activation function for theta.
                num_topics: Number of topics.


                id_embed_sc : Index map for the genes found in the Gene Expression (BoW) matrix input and the Knowledge Graph embedding genes. It should be a numpy list where each index maps to a gene in the embeddings. This allows aligning the relevant genes to the genes found in the embeddings.
                id_embed_ehr: Index map for the diseases found in the Electronic Health Record (BoW) matrix input and the Knowledge Graph embedding diseases. It should be a numpy list where each index maps to a disease in the embeddings. This allows aligning the relevant genes to the genes found in the embeddings.
                trainable_embeddings: Whether to fine-tune word embeddings.

                dropout: Dropout rate.

        """
        super(ETMModel, self).__init__()

        self.encoder_params = encoder_params

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
            return mu + eps * std
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

    def get_beta(self, modality: str):
        """
            Retrieve beta for the selecting modality which represents the topic-word (or topic-feature) distributions for that modality. It performs softmax of the vocabulary dimension. Calling this method puts the model into an evaluation state.

            Args:
                modality (str): "sc" single-cell RNA modality or "ehr" Electronic Health Record (diseases) modality.

            Returns:
                 np.ndarray: Beta representing the topic-word (or topic-feature) distributions.
        """
        if modality == 'sc':
            decoder = self.dec_sc
        elif modality == 'ehr':
            decoder = self.dec_ehr
        else:
            raise ValueError('The modality parameter must be either "sc" or "ehr".')

        with torch.no_grad():
            beta = decoder.get_beta().cpu().numpy()
        return beta

    def step_forward(self, encoder, decoder, bow, rho, aggregate=True):
        bow_raw  = bow # integer counts
        lengths  = bow_raw.sum(1, keepdim=True).clamp(min=1e-8)
        bow_norm = bow_raw / lengths # Normalize

        mu, logvar, kld = encoder(bow_norm) # NOTE: Is normalization appropriate (e.g.: for EHR)? Guess not..idk
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

    def forward(self, bow_sc, bow_ehr, rho_sc, rho_ehr):
        """
        Forward pass for the multi-modal Embedded Topic Model (ETM).
        Args:
            bow_sc : Bag-of-words representation of single-cell (SC) RNA modality.
            bow_ehr: Bag-of-words representation of Electronic Health Record (EHR) modality.
            rho_sc : Embedding rho for the single-cell (SC) RNA modality.
            rho_ehr: Embedding rho for the Electronic Health Record (EHR) modality.

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Tuple (outputs, losses) with outputs and losses
            dictionaries. Access the modality outputs with keys 'sc' or 'ehr' for either dictionary. Modalities within
            outputs contain keys 'theta' and 'preds'. Modalities within losses contain keys 'rec_loss' and 'kld'.

        """
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
                {'sc': losses_sc , 'ehr': losses_ehr})  # Losses