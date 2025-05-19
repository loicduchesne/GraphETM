# TRAINER IMPORTS
import os
import numpy as np

import torch
import torch.nn.functional as F

import wandb
from tqdm.notebook import tqdm, trange

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# TRAINER
class GraphETMTrainer:
    """
    Trains the dual Embedded Topic Model (ETM) Encoder and Decoder with the Graph Embeddings rho.

    Args:
        model: The GraphETM model.
        dataloader_sc (torch.utils.data.DataLoader): The dataloader for the scRNA BoW data. Rows are cells and columns are genes.
        dataloader_ehr (torch.utils.data.DataLoader): The dataloader for the EHR BoW data. Rows are patients and columns are ICD 9 codes.
        val_dataloader_sc (torch.utils.data.DataLoader): (Optional) The validation dataloader for the scRNA BoW data.
        val_dataloader_ehr (torch.utils.data.DataLoader): (Optional) The validation dataloader for the EHR BoW data.
        device (torch.device): Device to be used.
        wandb_run: (Optional) Wandb run object.
    """
    def __init__(self,
                 model,
                 dataloader_sc,
                 dataloader_ehr,
                 val_dataloader_sc=None,
                 val_dataloader_ehr=None,
                 device: torch.device = torch.device('cpu'),
                 wandb_run=None):

        # Objects
        self.model = model
        self.wandb = wandb_run
        self.device = device
        self.history = []

        self.dataloader_sc = dataloader_sc
        self.dataloader_ehr = dataloader_ehr
        self.val_dataloader_sc=val_dataloader_sc
        self.val_dataloader_ehr=val_dataloader_ehr

        self.proxy_sc = None
        self.proxy_ehr = None

    # ------------------------------------------------------------------
    def train(self,
              epochs,
              optimizer = None,
              verbose = True):
        """
        Train the model using the provided train_loaders and the specified num_epochs.

        Args:
            epochs (int): Number of epochs to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            verbose (bool): Disables training progress (e.g.: when optimizing hyperparameters with Optuna).
        """
        self.model.to(self.device)
        global_step = 0

        if self.wandb is not None:
            self.wandb.watch(
                self.model,
                log_freq=1,
                log='all'
            )

        pbar = trange(
            epochs,
            desc='Training GraphETM',
            unit='epoch',
            disable=not verbose,
            )

        # Two-phase training
        # for g in optimizer.param_groups:
        #     if g.get('name') in ('embedding_sc', 'embedding_ehr'):
        #         for p in g['params']:
        #             p.requires_grad_(False)

        for epoch in range(epochs):
            # pbar.set_description('Training GraphETM')
            pbar.update(1)
            # training ------------------------------------------------
            self.model.train()

            train_losses = {
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

            # Two-phase training
            # if epoch >= 10:
            #     for g in optimizer.param_groups:
            #         if g.get('name') in ('embedding_sc', 'embedding_ehr'):
            #             g['lr'] = g['lr']
            #             for p in g['params']:
            #                 p.requires_grad_(True)
            #
            #     self.model.dec_sc.embedding.requires_grad_(False)
            #     self.model.dec_ehr.embedding.requires_grad_(False)
            # else:
            #     self.model.dec_sc.embedding.requires_grad_(True)
            #     self.model.dec_ehr.embedding.requires_grad_(True)

            for bow_sc, bow_ehr in zip(self.dataloader_sc, self.dataloader_ehr):
                if bow_sc.shape[0] != bow_ehr.shape[0]:
                    continue

                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)
                optimizer.zero_grad()

                output = self.model.forward(bow_sc=bow_sc, bow_ehr=bow_ehr)

                loss = output['loss']
                loss.backward()
                optimizer.step()

                # Loss
                train_losses['loss'].append(loss.item())
                train_losses['sc']['rec_loss'].append(output['sc']['rec_loss'].item())
                train_losses['sc']['kl'].append(output['sc']['kl'].detach().cpu())
                train_losses['ehr']['rec_loss'].append(output['ehr']['rec_loss'].item())
                train_losses['ehr']['kl'].append(output['ehr']['kl'].detach().cpu())

                # Wandb (batch-level update)
                if self.wandb is not None:
                    self.wandb.log({
                        'batch': global_step,
                        'train/total_loss':     output['loss'].item(),
                        'train/sc/recon_loss' : output['sc']['rec_loss'].item(),
                        'train/sc/kld' :        float(output['sc']['kl'].detach().cpu()),
                        'train/ehr/recon_loss': output['ehr']['rec_loss'].item(),
                        'train/ehr/kld':        float(output['ehr']['kl'].detach().cpu()),
                    }, commit=True)
                global_step += 1


            # validation ------------------------------------------------
            val_loss = None
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

        pbar.close()

        # if self.wandb:
        #     self.wandb.finish()


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _evaluate(self, loader_sc, loader_ehr):
        self.model.eval()
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
            for bow_sc, bow_ehr in zip(loader_sc, loader_ehr):
                if bow_sc.shape[0] != bow_ehr.shape[0]:
                    continue
                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                output = self.model.forward(bow_sc=bow_sc, bow_ehr=bow_ehr)

                # Theta
                theta_sc_batches.append(output['sc']['theta'].cpu())
                theta_ehr_batches.append(output['ehr']['theta'].cpu())

                # Loss
                losses['loss'].append(output['loss'].item())
                losses['sc']['rec_loss'].append(output['sc']['rec_loss'].item())
                losses['sc']['kl'].append(output['sc']['kl'].item())
                losses['ehr']['rec_loss'].append(output['ehr']['rec_loss'].item())
                losses['ehr']['kl'].append(output['ehr']['kl'].item())

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
        beta_sc  = self.model.get_beta(modality='sc')
        beta_ehr = self.model.get_beta(modality='ehr')

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