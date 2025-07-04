### Imports
import torch
import torch.nn as nn
import torch.nn.functional as F


### GRAPH-ETM MODEL ARCHITECTURE
## ENCODER BLOCK
class Encoder(nn.Module):
    """
        Encoder module for GraphETM.

        Attributes:
                q_theta: q_theta
                theta_act: theta_act
                mu_q_theta: mu_q_theta
                logsigma_q_theta: logsigma_q_theta
    """
    def __init__(
            self,
            num_topics: int,
            vocab_size: int,
            encoder_hidden_size: int,
            dropout: float = 0.5,
            theta_act: str = 'tanh'
    ):
        """
            Initialize the Encoder module.

            Args:
                num_topics: Number of topics.
                vocab_size: Size of vocabulary.
                encoder_hidden_size: Size of the hidden layer in the encoder.
                theta_act: Activation function for theta.
        """
        super().__init__()

        # Dropout
        self.thres_dropout = dropout
        self.dropout = nn.Dropout(dropout)

        # Theta Activation
        self.theta_act = self._get_activation(theta_act)

        ## define variational distribution for \theta_{1:D} via amortization
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, encoder_hidden_size),
            self.theta_act,
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(encoder_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(encoder_hidden_size, num_topics, bias=True)

    def infer_topic_distribution(self, normalized_bows: torch.Tensor) -> torch.Tensor:
        """
            Returns a deterministic topic distribution for evaluation purposes bypassing the stochastic reparameterization step.

            Args:
                normalized_bows (torch.Tensor): Normalized bag-of-words input.

            Returns:
                torch.Tensor: Deterministic topic proportions.
        """
        q_theta = self.q_theta(normalized_bows)
        mu_theta = self.mu_q_theta(q_theta)
        theta = F.softmax(mu_theta, dim=-1)
        return theta

    def forward(self, bow_norm: torch.Tensor):
        """
        Returns parameters of the variational distribution for \theta.

        Args:
            bow_norm: (batch, V) Normalized batch of Bag-of-Words.

        Returns:
            mu_theta: mu_theta
            logsigma_theta: logsigma_theta
            kl_theta: kl_theta

        """
        q_theta = self.q_theta(bow_norm)
        if self.thres_dropout > 0:
            q_theta = self.dropout(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=1).mean()

        return mu_theta, logsigma_theta, kl_theta

    def _get_activation(self, act): # TODO: Redundant method.
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act