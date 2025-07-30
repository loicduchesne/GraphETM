### Imports
# Local
from .etm_model import ETMModel
from .graph_model import GraphModel
from .loss import GraphReconLoss
from .utils.run_helper import RunManager

# External
import numpy as np
from typing import Dict, Any
from itertools import cycle

import torch
import torch.nn.functional as F

import wandb
from tqdm.notebook import tqdm, trange

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


### GRAPHETM MODEL (TRAINER)
class GraphETM:
    """
    GraphETM model with an Embedded Topic Model (ETM) Encoder and Decoder and a Graph Convolutional Network (GCN) Filter.

    Args:
        etm_model_cfg (dict): The ETM model parameters.
        graph_model_cfg (dict): The Graph filter model parameters.
        graph_recon_loss_cfg (dict): The Graph Reconstruction Loss parameters.
        embedding (torch.Tensor): Initial embedding (also known as rho) computed from the knowledge graph (e.g.:
            TransE embeddings).
        edge_index (torch.LongTensor): The edge indices for the embedding matrix.
        id_embed_sc (np.ndarray): Array of indices corresponding to genes in the embedding matrix from the scRNA BoW
            data.
        id_embed_ehr (np.ndarray): Array of indices corresponding to diseases in the embedding matrix from the EHR BoW
            data.
        dataloader_sc (torch.utils.data.DataLoader): The dataloader for the scRNA BoW data. Rows are cells and columns
            are genes.
        dataloader_ehr (torch.utils.data.DataLoader): The dataloader for the EHR BoW data. Rows are patients and
            columns are ICD 9 codes.
        val_dataloader_sc (torch.utils.data.DataLoader, optional): The validation dataloader for the scRNA BoW data.
            Default: None.
        val_dataloader_ehr (torch.utils.data.DataLoader, optional): The validation dataloader for the EHR BoW data.
            Default: None.
        n_clusters_sc (int): Number of labeled clusters for the scRNA BoW data.
            Default: None.
        n_clusters_ehr (int): Number of labeled clusters for the EHR BoW data.
            Default: None.
        device (torch.device): Device to be used.
            Default: torch.device('cpu').
        wandb_run (optional): Wandb run object.
            Default: None.
    """
    def __init__(self,
                 # Models:
                 etm_model_cfg: Dict[str, Any],
                 graph_model_cfg: Dict[str, Any],
                 # Loss
                 graph_recon_loss_cfg: Dict[str, Any],
                 # Embedding:
                 embedding: torch.Tensor,
                 edge_index: torch.LongTensor,
                 id_embed_sc : np.ndarray,
                 id_embed_ehr: np.ndarray,
                 # Data:
                 dataloader_sc: torch.utils.data.DataLoader,
                 dataloader_ehr: torch.utils.data.DataLoader,
                 val_dataloader_sc: torch.utils.data.DataLoader = None,
                 val_dataloader_ehr: torch.utils.data.DataLoader = None,
                 # Labels:
                 n_clusters_sc : int = None,
                 n_clusters_ehr: int = None,
                 # Params:
                 device: torch.device = torch.device('cpu'),
                 wandb_run = None):

        ## Models
        self.etm_model = ETMModel(**etm_model_cfg, embedding_dim=graph_model_cfg['embedding_dim']).to(device) # TODO: Adjust model_cfg to account for rho
        self.graph_model = GraphModel(**graph_model_cfg, edge_index=edge_index.to(device), embedding=embedding).to(device)
        self.device = device

        # Graph Loss
        self.graph_recon_loss = GraphReconLoss(**graph_recon_loss_cfg).to(device)

        ## Graph Embeddings
        self.rho_full = embedding    # [N_total, L]
        self.edge_index = edge_index
        self.id_embed_sc  = torch.tensor(id_embed_sc , dtype=torch.long, device=device)
        self.id_embed_ehr = torch.tensor(id_embed_ehr, dtype=torch.long, device=device)

        ## Data
        self.dataloader_sc  = dataloader_sc
        self.dataloader_ehr = dataloader_ehr
        self.val_dataloader_sc  = val_dataloader_sc
        self.val_dataloader_ehr = val_dataloader_ehr

        ## Labels
        self.n_clusters_sc  = n_clusters_sc
        self.n_clusters_ehr = n_clusters_ehr

        ## ARI clustering
        self.proxy_sc = None
        self.proxy_ehr = None

        if wandb_run:
            self.run_manager = RunManager(wandb_run=wandb_run)

    # ------------------------------------------------------------------
    def train(self,
              epochs: int,
              optimizer: torch.optim.Optimizer,
              recon_loss_weight: float = 1.0,
              graph_loss_weight: float = 1.0,
              kld_max: float = 1.0,
              kld_annealing_duration: int = None,
              verbose: bool = True):
        """
        Train the model using the provided train_loaders and the specified num_epochs.

        Args:
            epochs (int): Number of epochs to train.
            optimizer (torch.optim.Optimizer): Optimizer to use.
            recon_loss_weight (float, optional): Weight for reconstruction loss.
                Default: 1.0.
            graph_loss_weight (float, optional): Weight for graph reconstruction loss.
                Default: 1.0.
            kld_max (float, optional): Maximum value for KL Divergence.
                Default: 1.0.
            kld_annealing_duration (int, optional): The duration (in epochs) for the KL Divergence to progressively
                reach its full magnitude. Default: None.
            verbose (bool, optional): Disables training progress (e.g.: when optimizing hyperparameters with Optuna).
                Default: True.
        """
        # Save configs
        self._recon_loss_weight = recon_loss_weight
        self._graph_loss_weight = graph_loss_weight
        self._kld_max = kld_max

        with self.run_manager.wandb_run(): # Catches failed runs
            self.etm_model.to(self.device), self.graph_model.to(self.device)

            if self.run_manager:
                self.run_manager.wandb.watch(self.etm_model, log_freq=1, log='all')
                self.run_manager.wandb.watch(self.graph_model, log_freq=1, log='all')

            pbar = trange(epochs, desc='Training GraphETM', unit='epoch', disable=not verbose)

            for epoch in range(epochs):
                pbar.update(1)
                # training ------------------------------------------------
                self.etm_model.train(), self.graph_model.train()

                # KL Annealing
                if kld_annealing_duration is not None:
                    kl_annealing = min(kld_max, epoch / kld_annealing_duration)
                else:
                    kl_annealing = 1.0

                # Data cycling (datasets different lengths)
                if len(self.dataloader_sc) > len(self.dataloader_ehr):
                    dataloader = zip(self.dataloader_sc, cycle(self.dataloader_ehr))
                else:
                    dataloader = zip(cycle(self.dataloader_sc), self.dataloader_ehr)

                for bow_sc, bow_ehr in dataloader:
                    if bow_sc.shape[0] != bow_ehr.shape[0]: # Exclude last batch
                        continue

                    # Zero grad
                    optimizer.zero_grad()

                    # To Device
                    bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                    # Filter: Forward
                    rho_full_new = self.graph_model.forward() # [N_total, L] # TODO: What about batching these, instead of full?
                    graph_loss = self.graph_recon_loss(rho_full_new, edge_index=self.edge_index) * graph_loss_weight

                    rho_sc  = rho_full_new[self.id_embed_sc ] # [V_sc , L]
                    rho_ehr = rho_full_new[self.id_embed_ehr] # [V_ehr, L]

                    # ETM: Forward
                    outputs, losses = self.etm_model(bow_sc=bow_sc, bow_ehr=bow_ehr, rho_sc=rho_sc, rho_ehr=rho_ehr)

                    # NELBO Loss
                    loss = ((losses['sc']['rec_loss'] + losses['ehr']['rec_loss']).mean()
                            + (losses['sc']['kld'] + losses['ehr']['kld']) * kl_annealing + graph_loss)
                    loss.backward()
                    optimizer.step()

                    # Log (batch-level) # TODO: Configure callbacks?
                    self.run_manager.log('total_loss', loss, on_step=True, phase='train')
                    self.run_manager.log('graph_recon_loss', graph_loss, on_step=True, phase='train')
                    self.run_manager.log_dict(losses, on_step=True, phase='train')
                    self.run_manager.next_step()

                # validation ------------------------------------------------
                if self.val_dataloader_sc:
                    self._evaluate(kl_annealing, recon_loss_weight, graph_loss_weight)

                self.run_manager.next_epoch()
            pbar.close()


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def _evaluate(self, kl_annealing, recon_loss_weight, graph_loss_weight):
        self.etm_model.eval(), self.graph_model.eval()

        with torch.no_grad():
            # Data cycling
            if len(self.val_dataloader_sc) > len(self.val_dataloader_ehr):
                dataloader = zip(self.val_dataloader_sc, cycle(self.val_dataloader_ehr))

            else:
                dataloader = zip(cycle(self.val_dataloader_sc), self.val_dataloader_ehr)

            theta_sc_batches  = []
            theta_ehr_batches = []
            labels_sc_batches  = [] # TODO: Make this method work if no labels are given.
            labels_ehr_batches = []
            for (bow_sc, labels_sc_batch), (bow_ehr, labels_ehr_batch) in dataloader: # TODO: Make this method work if no labels are given.
                if bow_sc.shape[0] != bow_ehr.shape[0]: # Exclude last batch
                    continue

                # Store labels
                labels_sc_batches.append(labels_sc_batch), labels_ehr_batches.append(labels_ehr_batch)

                # To Device
                bow_sc, bow_ehr = bow_sc.to(self.device), bow_ehr.to(self.device)

                # Graph: Forward
                rho_full_new = self.graph_model.forward() # [N_total, L]
                graph_loss = self.graph_recon_loss(rho_full_new, edge_index=self.edge_index) * graph_loss_weight

                rho_sc  = rho_full_new[self.id_embed_sc ] # [V_sc , L]
                rho_ehr = rho_full_new[self.id_embed_ehr] # [V_ehr, L]

                # ETM: Forward
                outputs, losses = self.etm_model(bow_sc=bow_sc, bow_ehr=bow_ehr, rho_sc=rho_sc, rho_ehr=rho_ehr)

                # ELBO Loss
                loss = (losses['sc']['rec_loss'] + losses['ehr']['rec_loss']).mean() + (losses['sc']['kld'] + losses['ehr']['kld']) * kl_annealing + graph_loss

                # Theta
                theta_sc_batches.append(outputs['sc']['theta'].cpu())
                theta_ehr_batches.append(outputs['ehr']['theta'].cpu())

                # Log (batch-level)
                self.run_manager.log('total_loss', loss, on_step=True, phase='val', aggregate=True)
                self.run_manager.log('graph_recon_loss', graph_loss, on_step=True, phase='val', aggregate=True)
                self.run_manager.log_dict(losses, on_step=True, phase='val', aggregate=True)

            ### METRICS
            theta_sc  = torch.cat(theta_sc_batches)
            theta_ehr = torch.cat(theta_ehr_batches)

            all_labels_sc = torch.cat(labels_sc_batches).numpy()
            all_labels_ehr = torch.cat(labels_ehr_batches).numpy()

            ari    = self.measure_ari(theta_sc, theta_ehr, all_labels_sc, all_labels_ehr)
            td     = self.topic_diversity()
            top_al = self.topic_embedding_alignment()

            # Log (epoch-level)
            self.run_manager.log_dict({'sc/ari': ari[0], 'ehr/ari': ari[1]}, on_epoch=True, phase='val')
            self.run_manager.log_dict({'sc/td': td[0], 'ehr/td': td[1]}, on_epoch=True, phase='val')
            self.run_manager.log('topic_emb_alignment', top_al, on_epoch=True, phase='val')


    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def measure_ari(self, theta_sc, theta_ehr, labels_sc, labels_ehr, random_state=0):
        """
        Measures the Adjusted Rand Index (ARI) score, using the labels given in __init__(). Higher is better,
        meaning cluster structures are stable. Lower values indicate that topics may have collapsed, rotated, or merged.

        Args:
            theta_sc: All of concatenated thetas Tensors for the scRNA modality.
            theta_ehr: All of concatenated thetas Tensors for the EHR modality.
            random_state: Internal KMeans clustering parameter.

        Returns:
            Tuple for the scRNA ARI (0) and the EHR ARI (1).
        """
        theta_sc  = theta_sc.numpy()
        theta_ehr = theta_ehr.numpy()

        if self.n_clusters_sc is None:
            self.n_clusters_sc = len(np.unique(labels_sc))
            self.n_clusters_ehr = len(np.unique(labels_ehr))

        pred_sc  = KMeans(n_clusters=self.n_clusters_sc , random_state=random_state).fit_predict(theta_sc)
        pred_ehr = KMeans(n_clusters=self.n_clusters_ehr, random_state=random_state).fit_predict(theta_ehr)

        ari_sc  = adjusted_rand_score(labels_sc , pred_sc)
        ari_ehr = adjusted_rand_score(labels_ehr, pred_ehr)

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

    def topic_embedding_alignment(self, top_k=15):
        """
        Check if topics learn similar patterns in embedding space
        """
        # Get average embeddings for top words in each topic
        beta_sc = self.etm_model.dec_sc.get_beta()
        beta_ehr = self.etm_model.dec_ehr.get_beta()

        topic_emb_sc = []
        topic_emb_ehr = []

        for k in range(beta_sc.shape[0]):
            # Top genes
            top_genes = beta_sc[k].topk(top_k).indices # Top genes for topic k
            gene_embs = self.graph_model.forward()[self.id_embed_sc[top_genes]]
            topic_emb_sc.append(gene_embs.mean(0))

            # Top diseases
            top_diseases = beta_ehr[k].topk(top_k).indices # Top diseases for topic k
            disease_embs = self.graph_model.forward()[self.id_embed_ehr[top_diseases]]
            topic_emb_ehr.append(disease_embs.mean(0))

        # Compare topic embeddings between modalities
        topic_emb_sc = torch.stack(topic_emb_sc)
        topic_emb_ehr = torch.stack(topic_emb_ehr)

        # Cosine similarity between corresponding topics
        cos_sim = F.cosine_similarity(topic_emb_sc, topic_emb_ehr)
        return cos_sim.mean().item()

    def measure_ari_proxy(self, theta_sc, theta_ehr, n_clusters=20, random_state=0):
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

    # @deprecated('This metric is not relevant since I am doing unpaired integration.')
    def ehr_from_scrna_recall(self, theta_sc, theta_ehr, k=5):
        th_sc  = F.normalize(theta_sc, p=2, dim=1)
        th_ehr = F.normalize(theta_ehr, p=2, dim=1)

        sims = th_sc @ th_ehr.T # cosine similarity
        topk = sims.topk(k, dim=1).indices
        rows = torch.arange(th_sc.size(0)).unsqueeze(1)
        return (topk == rows).any(1).float().mean().item()