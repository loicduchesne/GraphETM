# TRAINER IMPORTS
import os
import numpy as np

import torch
import wandb
from tqdm.notebook import tqdm, trange


# TRAINER
class GraphETMTrainer:
    """
    Trains the dual Embedded Topic Model (ETM) Encoder and Decoder with the Graph Embeddings rho.

    Args:
        model (pytorch_geometric.nn.Module): Ready-to-use nn.Module.
        train_dataloader (model.loader): A model.loader training dataloader object from a Pytorch Geometric KGE model.
        val_dataloader (model.loader): (Optional) A model.loader validation dataloader object from a Pytorch Geometric KGE model.
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

    # ------------------------------------------------------------------
    def train(self,
              epochs = 10,
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

            train_loss = np.mean(train_losses['loss'])

            # validation ------------------------------------------------
            val_loss = None
            val_losses = None
            if self.val_dataloader_sc:
                val_losses = self._evaluate(self.val_dataloader_sc, self.val_dataloader_ehr)

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
                        'val/ehr/kld':        np.mean(val_losses['ehr']['kl'])
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

            for bow_sc, bow_ehr in zip(loader_sc, loader_ehr):
                if bow_sc.shape[0] != bow_ehr.shape[0]:
                    continue
                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                output = self.model.forward(bow_sc=bow_sc, bow_ehr=bow_ehr)

                # Loss
                losses['loss'].append(output['loss'].item())
                losses['sc']['rec_loss'].append(output['sc']['rec_loss'].item())
                losses['sc']['kl'].append(output['sc']['kl'].item())
                losses['ehr']['rec_loss'].append(output['ehr']['rec_loss'].item())
                losses['ehr']['kl'].append(output['ehr']['kl'].item())
        return losses