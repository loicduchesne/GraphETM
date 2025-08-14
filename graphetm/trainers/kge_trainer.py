# TRAINER IMPORTS
import os
import numpy as np

import torch
import wandb
from tqdm.notebook import tqdm, trange


# TRAINER
class KGETrainer:
    """
    Trains a Knowledge Graph Embedding (KGE) PyTorch-Geometric model.

    Args:
        model (pytorch_geometric.nn.Module): Ready-to-use nn.Module.
        train_dataloader (model.loader): A model.loader training dataloader object from a Pytorch Geometric KGE model.
        val_dataloader (model.loader): (Optional) A model.loader validation dataloader object from a Pytorch Geometric KGE model.
        device (torch.device): Device to be used.
        wandb_run: (Optional) Wandb run object.
    """
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader=None,
                 device: torch.device = torch.device('cpu'),
                 wandb_run=None):

        # Objects
        self.model = model
        self.wandb = wandb_run
        self.device = device
        self.history = []

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

    # ------------------------------------------------------------------
    def _make_loader(self, triples_data, batch_size, shuffle=True):
        return self.model.loader(
            triples_data.edge_index[0],
            triples_data.edge_type,
            triples_data.edge_index[1],
            batch_size=batch_size,
            shuffle=shuffle)


    def fit(self,
            epochs = 10,
            optimizer = None,
            verbose = True):

        self.model.to(self.device)

        pbar = trange(
            epochs*(len(self.train_loader)),
            desc=f'Epoch 0: Training',
            unit='batch',
            disable=not verbose,
            )

        for epoch in range(epochs):
            pbar.set_description(f'Epoch {epoch}: Training KGE')
            # training ------------------------------------------------
            self.model.train()
            train_losses = []
            for h, r, t in self.train_loader: # (head_index, relation_type, tail_index)
                pbar.update(1)

                h, r, t = h.to(self.device), r.to(self.device), t.to(self.device)
                optimizer.zero_grad()

                loss = self.model.loss(h, r, t)
                loss.backward()
                optimizer.step()

                # Loss
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            self.history.append(train_losses) # Append entire loss history.

            # validation ------------------------------------------------
            val_loss = None
            val_losses = None
            if self.val_loader:
                pbar.set_description(f'Epoch {epoch}: Validation')

                val_losses = self._evaluate(self.val_loader)
                val_loss = np.mean(val_losses)

            self.history.append((epoch, train_losses, val_losses))

            if self.wandb:
                self.wandb.log({'train/loss': train_loss, 'val/loss': val_loss}, step=epoch)

        pbar.close()

        if self.wandb:
            self.wandb.finish()


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _evaluate(self, loader):
        self.model.eval()
        with torch.no_grad():
            losses = []
            for h, r, t in loader:
                h, r, t = h.to(self.device), r.to(self.device), t.to(self.device)

                loss = self.model.loss(h, r, t)

                # Loss
                losses.append(loss.item())
        return losses