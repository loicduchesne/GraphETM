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
# Description: MODEL assembles the GraphETM architecture using the ENCODER, DECODER, and GRAPH_CONV_FILTER blocks.

# ------------------------------------------------------------------
# @title GraphETM Model # TODO: I could potentially rename this etm_model (and the gcn graph_model) for consistency.
class Model(nn.Module):
    def __init__(
            self,
            encoder_params  : Dict[str, Dict[str, int]],
            theta_act: str,
            embedding_dim : int,
            num_topics: int,
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
        super(Model, self).__init__()

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
        theta = self.encoder.infer_topic_distribution(normalized_bows) # TODO: To fix.
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

    def graph_forward(self):
        pass #  Maybe not idk

    def step_forward(self, encoder, decoder, bow, rho, aggregate=True):
        bow_raw  = bow # integer counts
        lengths  = bow_raw.sum(1, keepdim=True) + 1e-8
        bow_norm = bow_raw / lengths # Normalize

        mu, logvar, kld = encoder(bow_norm)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=-1) # D x K (batch_size x num_topics)

        preds = decoder(theta, rho=rho) # TODO: Shouldn't preds be the same shape as bow?
        rec_loss = -(preds * bow_raw).sum(1) #/ lengths.squeeze(1) # Dev. note: lengths.squeeze(1) is the only key difference.
        if aggregate:
            rec_loss = rec_loss.mean()

        return {
            'rec_loss': rec_loss,
            'kl'      : kld,
            'theta'   : theta.detach(),
            'preds'   : preds.detach(),
        }

    def forward(self, bow_sc, bow_ehr, rho_sc, rho_ehr, kl_annealing=1.0):
        # Encoder-Decoder: ScRNA
        output_sc = self.step_forward(
            bow=bow_sc,
            encoder=self.enc_sc,
            decoder=self.dec_sc,
            rho=rho_sc)

        # Encoder-Decoder: EHR
        output_ehr = self.step_forward(
            bow=bow_ehr,
            encoder=self.enc_ehr,
            decoder=self.dec_ehr,
            rho=rho_ehr)


        return {
            'sc' : output_sc,
            'ehr': output_ehr,
        }