# Graph-ETM: Graph-Embedded Topic Model for diagonal integration for Single-Cell RNA-seq and Electronic Health Records data

*Supervised research project done at McGill University.*

> [!WARNING]
> Project is still actively in development, and the model architecture may contain errors.

GraphETM is a PyTorch-based model integrating knowledge graph embeddings into an Embedded Topic Model (ETM: ref) with multi-modality integration of both single-cell RNA-seq (gene expression) data and electronic health records data.

## Usage
This section quickly goes over the project structure and usage.

### Step 1: Data pre-processing
- **Run `data.ipynb` :** This notebook processes the raw data to be used for the model. It only works for the specific iBKH, MIMIC-III, and PBMC datasets. It expects the following file structure:

```bash
├── data
│   ├── iBKH
│   │   ├── D_D_res.csv
│   │   ├── D_Di_res.csv
│   │   ├── D_G_res.csv
│   │   ├── Di_Di_res.csv
│   │   ├── Di_G_res.csv
│   │   ├── G_G_res.csv
│   │   ├── disease_vocab.csv
│   │   ├── drug_vocab.csv
│   │   └── gene_vocab.csv
│   ├── MIMIC-III
│   │   ├── D_ICD_DIAGNOSES.csv
│   │   └── DIAGNOSES_ICD.csv
│   └── PBMC
│       ├── barcodes.tsv
│       ├── genes.tsv
│       └── matrix.mtx
├── inputs
│   ├── GraphETM
│   │   └── *empty*
│   └── KGE
│       └── *empty*
├── data.ipynb
...
```
Where the `data/...` contains the *raw* data. The folder names are case-sensitive. The `inputs/...` folder contains the *processed* data after `data.ipynb` has been successfully run.

Afterward, the model scripts will use the processed files found in the `inputs/...` folder.

### Step 2: Generating the Graph Embeddings
> [!IMPORTANT]
> This notebook uses Weight & Biases[^1] (Wandb) to monitor training metrics such as loss, ARI, etc. I highly recommend logging in a Wandb account to properly use these notebooks.

- **Run `KGE.ipynb` :** To generate the knowledge graph embeddings, we use the TransE embedding model from PyTorch-Geometric (ref) and the custom trainer specified in `model/kge_trainer.py`.

The **data is first prepared** using a custom dataset defined in `utils/datasets.py`, which will generate triplets saved in `inputs/KGE/input_triplet.pt`. Running `KGE.ipynb` already has the triplet generation code included. It will use TransE to generate the embeddings which are then saved in `inputs/GraphETM/embedding_full.pt`[^2].

[^1]: [Link to Weight & Biases.](https://wandb.ai/site/)

[^2]: In order to align the embeddings with the input modalities datasets (MIMIC-III and PBMC), `inputs/GraphETM/id_embed_ehr.npy` and `inputs/GraphETM/id_embed_sc.npy` are also generated in the process.

### Step 3: Running GraphETM
> [!IMPORTANT]
> This notebook uses Weight & Biases[^1] (Wandb) to monitor training metrics such as loss, ARI, etc. I highly recommend logging in a Wandb account to properly use these notebooks.

- **Run `GraphETM.ipynb` :** Finally, **once the data and the embeddings have been prepared** we can run the dual-modality ETM model. It uses the architecture defined within the notebook and a custom trainer written in `model.graphetm_trainer.py`. The results can be visualized using the various plots in the notebook.