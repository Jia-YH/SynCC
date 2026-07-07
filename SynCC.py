"""SynCC -- main entry point.

Single-cell RNA-seq clustering & imputation via a GAT masked graph
auto-encoder with contrastive and ZINB objectives.

Reorganized from the original ``Test_GCMA.py`` (clustering / ``cls`` task).

Usage
-----
Train and evaluate::

    python SynCC.py --dataset baron_mouse --name baron_mouse --task cls \
        --use_cfg --scheduler --save_model --model_path checkpoints/baron_mouse.pt

Evaluate a previously trained checkpoint (unchanged precision)::

    python SynCC.py --dataset Quake_Muscle --name Quake_Muscle --task cls \
        --use_cfg --load_model --model_path finalmodel_editDM/last/Quake_Muscle_up.pt
"""
import os
import logging

import numpy as np
import torch
from tqdm import tqdm

from config import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    load_best_configs,
)
from data import Trainer
from model import build_model
from evaluation import clustering_for_transductive, imputation_error

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler,
             num_classes, cell_type, X_raw, drop_index, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat

    epoch_iter = tqdm(range(max_epoch))
    x_rec = None

    for epoch in epoch_iter:
        model.train()
        loss, loss_dict, loss_components, x_rec, mean, disp, pi = model(graph, x, cell_type, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if (epoch + 1) % 20 == 0:
            clustering_for_transductive(model, graph, x, num_classes, cell_type, x_rec)
            rmse, median_l1_distance, cosine_similarity = imputation_error(x_rec, X_raw, drop_index)
            print("rmse median_l1_distance cosine_similarity", rmse, median_l1_distance, cosine_similarity)

    return model, x_rec


def main(args):
    device = 'cuda'
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    logs = args.logging
    use_scheduler = args.scheduler

    # ---- data & graph ----
    embedder = Trainer(args)
    graph, num_features, features_train, cell_type, num_classes, X_raw, drop_index = embedder.data()
    cell_type = torch.FloatTensor(cell_type).to(device)
    args.num_features = num_features

    nmi_list, cls_acc_list, ari_list, ca_list = [], [], [], []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_wd_{weight_decay}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args, num_features, num_classes)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        else:
            scheduler = None

        x = torch.FloatTensor(features_train).to(device)

        if not args.load_model:
            model, x_rec = pretrain(
                model, graph, x, optimizer, max_epoch, device, scheduler,
                num_classes, cell_type, X_raw, drop_index, logger)
            model = model.cpu()

        if args.load_model:
            logging.info(f"Loading Model from {args.model_path} ... ")
            model.load_state_dict(torch.load(args.model_path))

        if args.save_model:
            logging.info(f"Saving Model to {args.model_path} ...")
            os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.model_path)

        model = model.to(device)
        model.eval()

        # ---- evaluation (clustering + imputation) ----
        graph = graph.to(device)
        x = x.to(device)
        x_rec = None
        final_nmi, final_acc, ari, ca = clustering_for_transductive(
            model, graph, x, num_classes, cell_type, x_rec)
        rmse, median_l1_distance, cosine_similarity = imputation_error(x_rec, X_raw, drop_index)
        print("rmse median_l1_distance cosine_similarity", rmse, median_l1_distance, cosine_similarity)

        nmi_list.append(final_nmi)
        cls_acc_list.append(final_acc)
        ari_list.append(ari)
        ca_list.append(ca)

        if logger is not None:
            logger.finish()

    final_nmi, final_nmi_std = np.mean(nmi_list), np.std(nmi_list)
    final_ari, final_ari_std = np.mean(ari_list), np.std(ari_list)
    final_ca, final_ca_std = np.mean(ca_list), np.std(ca_list)
    print(f"# final_nmi: {final_nmi:.4f}±{final_nmi_std:.4f}")
    print(f"# final_ari: {final_ari:.4f}±{final_ari_std:.4f}")
    print(f"# final_ca: {final_ca:.4f}±{final_ca_std:.4f}")
    print(nmi_list)
    print(final_ari)


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs.yml")
    print(args)

    main(args)
