### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
        Decoder module for GraphETM.

        Attributes:
            rho: [V (Vocab. size), L (Emb. dim.)] Word embedding matrix.
            alphas: [L, K (topic size)] Topic embedding matrix.
    """
    def __init__(
            self,
            embedding_dim: int,
            num_topics: int,
    ):
        """
            Initialize the Decoder module.

            Args:
                num_topics: Number of topics.

        """
        super().__init__()

        # Objective: The latent topic distribution theta for (scRNA and EHR) are multiplied with Beta (essentially grounding the latent topics with the knowledge).

        ## define the word embedding matrix \rho
        self.rho = None # [V, L]

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(embedding_dim, num_topics, bias=False)

        ## define beta
        self.beta = None

    def get_beta(self):
        """
            Retrieve beta for a modality representing the topic-word (or topic-feature) distributions. It performs softmax of the vocabulary dimension.

            Returns:
                Beta which represents the topic-word (or topic-feature) distributions.
        """
        logits = self.alphas(self.rho)
        self.beta = F.softmax(logits.T, dim=1) # NOTE: Transposing here now for fun i guess.
        return self.beta

    def forward(self, theta, rho):
        self.rho = rho # Update embeddings

        beta = self.get_beta()
        preds = torch.log(torch.mm(theta, beta) + 1e-8)
        return preds