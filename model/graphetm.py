### Imports
# Local
from .model import Model
from .conv_filter import GraphFilter
from .loss import graph_recon_loss

# External
import numpy as np
from itertools import cycle

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

import wandb
from tqdm.notebook import tqdm, trange

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# TRAINER
class GraphETM:
    """
    GraphETM model  Embedded Topic Model (ETM) Encoder and Decoder with the Graph Embeddings rho.

    Args:
        model: The GraphETM model.

        embedding: Initial embedding (also known as rho) computed from the knowledge graph (e.g.: TransE embeddings).

        dataloader_sc (torch.utils.data.DataLoader): The dataloader for the scRNA BoW data. Rows are cells and columns are genes.
        dataloader_ehr (torch.utils.data.DataLoader): The dataloader for the EHR BoW data. Rows are patients and columns are ICD 9 codes.
        val_dataloader_sc (torch.utils.data.DataLoader): (Optional) The validation dataloader for the scRNA BoW data.
        val_dataloader_ehr (torch.utils.data.DataLoader): (Optional) The validation dataloader for the EHR BoW data.
        device (torch.device): Device to be used.
        wandb_run: (Optional) Wandb run object.
    """
    def __init__(self,
                 # Models:
                 model_cfg,
                 gcn_cfg,
                 # Embedding:
                 embedding: torch.Tensor,
                 graphloader_cfg: dict,
                 edge_index: torch.LongTensor,
                 id_embed_sc : np.ndarray,
                 id_embed_ehr: np.ndarray,
                 # Data:
                 dataloader_sc,
                 dataloader_ehr,
                 val_dataloader_sc=None,
                 val_dataloader_ehr=None,
                 # Params:
                 trainable_embeddings=False, # Must be false to keep embeddings stable.
                 device: torch.device=torch.device('cpu'),
                 wandb_run=None):

        ## Models
        self.etm_model = Model(**model_cfg, embedding_dim=gcn_cfg['embedding_dim']).to(device) # TODO: Adjust model_cfg to account for rho
        self.graph_model = GraphFilter(**gcn_cfg, in_dim=embedding.shape[1], edge_index=edge_index).to(device)
        self.device = device

        ## Graph Embeddings
        self.rho_full = embedding # V x L
        self.edge_index = edge_index
        self.id_embed_sc  = torch.tensor(id_embed_sc , dtype=torch.long, device=device)
        self.id_embed_ehr = torch.tensor(id_embed_ehr, dtype=torch.long, device=device)

        self.base_rho_sc  = embedding[id_embed_sc ].to(device)   # [V_sc , L]
        self.base_rho_ehr = embedding[id_embed_ehr].to(device)   # [V_ehr, L]

        # global2vocab map for indexing
        # -1 for non-modality nodes
        self.global2vocab_sc = -torch.ones(embedding.size(0), dtype=torch.long, device=device)
        self.global2vocab_sc[self.id_embed_sc] = torch.arange(self.id_embed_sc.size(0), dtype=torch.long, device=device)

        self.global2vocab_ehr = -torch.ones(embedding.size(0), dtype=torch.long, device=device)
        self.global2vocab_ehr[self.id_embed_ehr] = torch.arange(self.id_embed_ehr.size(0), dtype=torch.long, device=device)


        self.dataloader_emb = NeighborLoader(
            **graphloader_cfg,
            data=Data(x=embedding, edge_index=edge_index),
            input_nodes=None,
        )

        self._emb_iter = cycle(self.dataloader_emb)
        self._emb_iter_val = cycle(self.dataloader_emb)

        ## Data
        self.dataloader_sc  = dataloader_sc
        self.dataloader_ehr = dataloader_ehr
        self.val_dataloader_sc  = val_dataloader_sc
        self.val_dataloader_ehr = val_dataloader_ehr

        self.proxy_sc  = None
        self.proxy_ehr = None

        self.wandb = wandb_run
        self.history = []

    # ------------------------------------------------------------------
    def train(self,
              epochs,
              optimizer = None,
              kl_annealing_duration = None,
              verbose = True):
        """
        Train the model using the provided train_loaders and the specified num_epochs.

        Args:
            epochs (int): Number of epochs to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            kl_annealing_duration (float): The duration (in epochs) for the KL Divergence to progressively reach its full magnitude.
            verbose (bool): Disables training progress (e.g.: when optimizing hyperparameters with Optuna).
        """
        self.etm_model.to(self.device), self.graph_model.to(self.device)
        global_step = 0

        if self.wandb is not None:
            self.wandb.watch(
                self.etm_model,
                log_freq=1,
                log='all'
            )

        pbar = trange(
            epochs,
            desc='Training GraphETM',
            unit='epoch',
            disable=not verbose,
            )

        # KL Annealing
        kl_annealing = 1.0
        kl_annealing_coef = 0.0
        if kl_annealing_duration is not None:
            kl_annealing = 0.0
            kl_annealing_coef = 1.0 / kl_annealing_duration

        for epoch in range(epochs):
            pbar.update(1)
            # training ------------------------------------------------
            self.etm_model.train()
            self.graph_model.train()

            train_losses = {
                'loss': [],
                'graph_rec_loss': [],
                'sc': {
                    'rec_loss': [],
                    'kl'      : [],
                },
                'ehr': {
                    'rec_loss': [],
                    'kl'      : [],
                }
            }

            for bow_sc, bow_ehr in zip(self.dataloader_sc, self.dataloader_ehr):
                g = next(self._emb_iter)

                if bow_sc.shape[0] != bow_ehr.shape[0]: # Exclude last batch
                    continue

                # Zero grad
                optimizer.zero_grad()

                # To Device
                g = g.to(self.device)
                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                # Graph: Forward
                h = self.graph_model(g.x, g.edge_index)
                graph_loss = graph_recon_loss(h, g.edge_index)

                mask_sc  = torch.isin(g.n_id, self.id_embed_sc)
                mask_ehr = torch.isin(g.n_id, self.id_embed_ehr)

                # Local positions in the subgraph
                loc_sc  = mask_sc.nonzero(as_tuple=False).squeeze() # Should be 71
                loc_ehr = mask_ehr.nonzero(as_tuple=False).squeeze()

                # global IDs
                global_sc  = g.n_id[loc_sc] # Should be 71
                global_ehr = g.n_id[loc_ehr]

                # map global IDs to row slots in modality vocab
                row_sc  = self.global2vocab_sc[global_sc]
                row_ehr = self.global2vocab_ehr[global_ehr]

                rho_sc_full = self.base_rho_sc.clone()   # [V_sc, L]
                rho_sc_full[row_sc] = h[loc_sc]

                rho_ehr_full = self.base_rho_ehr.clone() # V [V_ehr, L]
                rho_ehr_full[row_ehr] = h[loc_ehr]

                # ETM: Forward
                outputs = self.etm_model(bow_sc=bow_sc, bow_ehr=bow_ehr, rho_sc=rho_sc_full, rho_ehr=rho_ehr_full, kl_annealing=kl_annealing)

                # ELBO Loss
                loss = (outputs['sc']['rec_loss'] + outputs['ehr']['rec_loss']).mean() + (outputs['sc']['kl'] + outputs['ehr']['kl']) * kl_annealing + graph_loss
                loss.backward()
                optimizer.step()

                # Save Loss # TODO: Clean this mess.
                train_losses['loss'].append(loss.item())
                train_losses['sc']['rec_loss'].append(outputs['sc']['rec_loss'].mean().item())
                train_losses['sc']['kl'].append(outputs['sc']['kl'].detach().cpu())
                train_losses['ehr']['rec_loss'].append(outputs['ehr']['rec_loss'].mean().item())
                train_losses['ehr']['kl'].append(outputs['ehr']['kl'].detach().cpu())

                # Wandb (batch-level update)
                if self.wandb is not None:
                    self.wandb.log({
                        'batch': global_step,
                        'train/total_loss':       loss.item(),
                        'train/graph_recon_loss': graph_loss.item(),
                        'train/sc/recon_loss' :   outputs['sc']['rec_loss'].item(),
                        'train/sc/kld' :          float(outputs['sc']['kl'].detach().cpu()),
                        'train/ehr/recon_loss':   outputs['ehr']['rec_loss'].item(),
                        'train/ehr/kld':          float(outputs['ehr']['kl'].detach().cpu()),
                    }, commit=True)
                global_step += 1


            # validation ------------------------------------------------
            val_losses = None
            metrics = None
            if self.val_dataloader_sc:
                val_losses, metrics = self._evaluate(self.val_dataloader_sc, self.val_dataloader_ehr)

            self.history.append((epoch, train_losses, val_losses))

            # Wandb (epoch-level update)
            if self.wandb is not None:
                self.wandb.log(
                    data={
                        # Validation
                        'epoch': epoch,
                        'val/total_loss':     np.mean(val_losses['loss']),
                        'val/sc/recon_loss' : np.mean(val_losses['sc']['rec_loss']),
                        'val/sc/kld' :        np.mean(val_losses['sc']['kl']),
                        'val/ehr/recon_loss': np.mean(val_losses['ehr']['rec_loss']),
                        'val/ehr/kld':        np.mean(val_losses['ehr']['kl']),
                        # Metrics
                        'val/sc/ari' : metrics['sc' ]['ari'],
                        'val/ehr/ari': metrics['ehr']['ari'],
                        'val/sc/td' : metrics['sc' ]['td'],
                        'val/ehr/td': metrics['ehr']['td'],
                        'val/recall@5': metrics['recall@5'],
                        },
                    commit=True)

            # KL Annealing Update
            kl_annealing += kl_annealing_coef

        pbar.close()


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _evaluate(self, loader_sc, loader_ehr):
        self.etm_model.eval()
        self.graph_model.eval()

        with torch.no_grad():
            losses = {
                'loss': [],
                'sc': {
                    'rec_loss': [],
                    'kl'      : [],
                },
                'ehr': {
                    'rec_loss': [],
                    'kl'      : [],
                }
            }
            metrics = {
                'sc' : {},
                'ehr': {},
                'recall@5': 0
            }

            theta_sc_batches  = []
            theta_ehr_batches = []
            for bow_sc, bow_ehr in zip(loader_sc, loader_ehr): # TODO: (DONE) Batch embedding here as well (although because of nograd maybe i can do the full)
                g = next(self._emb_iter_val)

                if bow_sc.shape[0] != bow_ehr.shape[0]: # Exclude last batch
                    continue

                # To Device
                g = g.to(self.device)
                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                # Graph: Forward
                h = self.graph_model(g.x, g.edge_index)
                graph_loss = graph_recon_loss(h, g.edge_index)

                mask_sc  = torch.isin(g.n_id, self.id_embed_sc)
                mask_ehr = torch.isin(g.n_id, self.id_embed_ehr)

                # Local positions in the subgraph
                loc_sc  = mask_sc.nonzero(as_tuple=False).squeeze() # Should be 71
                loc_ehr = mask_ehr.nonzero(as_tuple=False).squeeze()

                # global IDs
                global_sc  = g.n_id[loc_sc] # Should be 71
                global_ehr = g.n_id[loc_ehr]

                # map global IDs to row slots in modality vocab
                row_sc  = self.global2vocab_sc[global_sc]
                row_ehr = self.global2vocab_ehr[global_ehr]

                rho_sc_full = self.base_rho_sc.clone()   # [V_sc, L]
                rho_sc_full[row_sc] = h[loc_sc]

                rho_ehr_full = self.base_rho_ehr.clone() # V [V_ehr, L]
                rho_ehr_full[row_ehr] = h[loc_ehr]

                # ETM: Forward
                outputs = self.etm_model(bow_sc=bow_sc, bow_ehr=bow_ehr, rho_sc=rho_sc_full, rho_ehr=rho_ehr_full)

                # ELBO Loss
                loss = (outputs['sc']['rec_loss'] + outputs['ehr']['rec_loss']).mean() + outputs['sc']['kl'] + outputs['ehr']['kl'] + graph_loss

                # Theta
                theta_sc_batches.append(outputs['sc']['theta'].cpu())
                theta_ehr_batches.append(outputs['ehr']['theta'].cpu())

                # Loss
                losses['loss'].append(loss.item())
                losses['sc']['rec_loss'].append(outputs['sc']['rec_loss'].mean().item())
                losses['sc']['kl'].append(outputs['sc']['kl'].item())
                losses['ehr']['rec_loss'].append(outputs['ehr']['rec_loss'].mean().item())
                losses['ehr']['kl'].append(outputs['ehr']['kl'].item())

            ### METRICS
            theta_sc  = torch.cat(theta_sc_batches)
            theta_ehr = torch.cat(theta_ehr_batches)

            metrics['sc']['ari'], metrics['ehr']['ari'] = self.measure_ari(theta_sc, theta_ehr)
            metrics['sc']['td'], metrics['ehr']['td'] = self.topic_diversity()
            metrics['recall@5'] = self.ehr_from_scrna_recall(theta_sc, theta_ehr)

        return losses, metrics


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def measure_ari(self, theta_sc, theta_ehr, n_clusters=20, random_state=0):
        """
        Measures the Adjusted Rand Index (ARI) score, using proxy clusters fixed at the first epoch. Higher is better,
        meaning cluster structures are stable. Lower values indicate that topics may have collapsed, rotated, or merged.

        Args:
            theta_sc: All of concatenated thetas Tensors for the scRNA modality.
            theta_ehr: All of concatenated thetas Tensors for the EHR modality.
            n_clusters: Internal KMeans clustering parameter.
            random_state: Internal KMeans clustering parameter.

        Returns:
            Tuple for the scRNA ARI (0) and the EHR ARI (1).
        """
        theta_sc  = theta_sc.numpy()
        theta_ehr = theta_ehr.numpy()

        if self.proxy_sc is None:
            self.proxy_sc  = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(theta_sc)
            self.proxy_ehr = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(theta_ehr)

            ari_sc  = None
            ari_ehr = None
        else:
            pred_sc  = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(theta_sc)
            pred_ehr = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(theta_ehr)

            ari_sc  = adjusted_rand_score(self.proxy_sc,  pred_sc)
            ari_ehr = adjusted_rand_score(self.proxy_ehr, pred_ehr)

        return ari_sc, ari_ehr

    def topic_diversity(self, top_k=15):
        """
        Measures the fraction of unique tokens among the top-k words of every topic. Higher values indicate more diverse
        topic vocabulary. Lower values indicate that a few tokens dominate every topic.

        Args:
            top_k: How many words to consider per topic.

        Returns:
            Tuple for the scRNA TD (0) and the EHR TD (1).
        """
        beta_sc  = self.etm_model.get_beta(modality='sc')
        beta_ehr = self.etm_model.get_beta(modality='ehr')

        top_sc  = np.argsort(beta_sc,  axis=1)[:, -top_k:]
        top_ehr = np.argsort(beta_ehr, axis=1)[:, -top_k:]

        td_sc  = np.unique(top_sc ).size / (beta_sc.shape[0]  * top_k)
        td_ehr = np.unique(top_ehr).size / (beta_ehr.shape[0] * top_k)

        return td_sc, td_ehr

    def ehr_from_scrna_recall(self, theta_sc, theta_ehr, k=5):
        th_sc  = F.normalize(theta_sc, p=2, dim=1)
        th_ehr = F.normalize(theta_ehr, p=2, dim=1)

        sims = th_sc @ th_ehr.T # cosine similarity
        topk = sims.topk(k, dim=1).indices
        rows = torch.arange(th_sc.size(0)).unsqueeze(1)
        return (topk == rows).any(1).float().mean().item()