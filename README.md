# SynCC

SynCC: Synergistic Learning for Robust Clustering in Single-Cell Transcriptomics

## 📖 Overview
**SynCC** is a self-supervised deep learning framework for accurate cell type and state identification from scRNA-seq data.  
It features a **primary–auxiliary modular design** that synergizes:
- **Hierarchical Contrasting**   captures global intercellular relationships via graph contrastive learning  
- **Integrated Reconstruction**  refines local representations through a regulatory autoencoding pathway  
- **Dynamic Augmentation**  perturbs data proportional to cellular importance, preserving heterogeneity  

On six benchmark datasets, SynCC outperforms existing methods by up to **22.4% (ARI)**, **8.9% (NMI)**, and **12.3% (ACC)**, while also excelling in biological interpretability (marker gene identification).

## 📂 Repository Structure

| File | Responsibility |
|------|----------------|
| `SynCC.py` | Main entry point (train / evaluate, `cls` task) |
| `config.py` | Argument parsing, optimizer/seed/norm helpers, `TBLogger` |
| `data.py` | `.h5ad` loading, preprocessing, dropout mask, kNN graph, `Trainer` |
| `graph.py` | `knn_graph` cell-graph builder |
| `model.py` | `GAT`, `setup_module`, `PreModel`, `build_model` |
| `backbones.py` | SAGE / GIN / GCN / DotGAT backbones (non-default) |
| `layers.py` | `ZINBLoss`, `MeanAct`, `DispAct` |
| `losses.py` | `sce_loss`, `sig_loss` |
| `graph_ops.py` | Augmentation / centrality drop operators |
| `evaluation.py` | Clustering (NMI/ARI/CA) + imputation (RMSE/L1/cos) metrics |
| `configs.yml` | Per-dataset best hyper-parameters |

## Usage

The `dataset/` directory (with the `.h5ad` files) and, for evaluation, the
trained checkpoints must be present relative to the working directory.

Train + evaluate:

```bash
python SynCC.py --dataset Quake_Muscle --name Quake_Muscle\
    --use_cfg --scheduler --save_model --model_path checkpoints/Quake_Muscle.pt
```

Evaluate an existing checkpoint (identical precision to the original):

```bash
python SynCC.py --dataset Quake_Muscle --name Quake_Muscle\
    --use_cfg --load_model --model_path finalmodel_editDM/last/Quake_Muscle_up.pt
```

Notes:
- `--use_cfg` loads per-dataset hyper-parameters from `configs.yml`.
- `--name` selects the `.h5ad` file (`dataset/{name}.h5ad`); `--dataset`
  selects the config block. They are usually the same.

