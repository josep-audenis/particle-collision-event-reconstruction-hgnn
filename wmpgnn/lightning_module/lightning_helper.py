import pdb
import re

import torch

import pandas as pd


def make_loggable(pos_weight_dict):
    loggable = {}
    for k, v in pos_weight_dict.items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                loggable[k] = v.item()
            else:
                loggable[k] = v.tolist()
        else:
            loggable[k] = str(v)
    return loggable


def init_logs(configs, mode="train"):
    model = configs["model"]
    loss_config = configs["inference"]

    log = {}
    if model == "DFEI":
        log["combined_loss"] = []
        gn_blocks = configs[model]["GNblocks"]["nBlocks"]
        if loss_config["LCA"]:
            log["LCA_loss"] = []
            for i in range(4):
                log[f"LCA_class{i}_num"] = []
                for j in range(4):
                    log[f"LCA_class{i}_pred_class{j}"] = []

        if loss_config["node_prune"]:
            log["t_nodes_loss"] = []
            if mode == "test":
                for i in range(gn_blocks):
                    log[f"sig_nodes_score_{i}"] = torch.tensor([], dtype=torch.float16)
                    log[f"bkg_nodes_score_{i}"] = torch.tensor([], dtype=torch.float16)

        if loss_config["edge_prune"]:
            log["tt_edges_loss"] = []
            if mode == "test":
                for i in range(gn_blocks):
                    log[f"sig_edges_score_{i}"] = torch.tensor([], dtype=torch.float16)
                    log[f"bkg_edges_score_{i}"] = torch.tensor([], dtype=torch.float16)

        if  loss_config["pv_asso"]:
            log["tpv_edges_loss"] = []
            if mode == "test":
                for i in range(gn_blocks):
                    log[f"sig_pv_asso_score_{i}"] = torch.tensor([], dtype=torch.float16)
                    log[f"bkg_pv_asso_score_{i}"] = torch.tensor([], dtype=torch.float16)


    elif model == "IFT":
        if loss_config["FT"]:
            log["ft_loss"] = []
            if mode == "train":
                log["ft_auc"] = []
                log["ft_tagging_power_proxy"] = []
                log["ft_wrong_fraction_proxy"] = []
    if mode == "train":
        return log, log
    elif mode == "test":
        return log
    else:
        raise Exception(f"Unknown mode {mode}")


def init_loss(device):
    loss = {"LCA": torch.tensor(0., device=device),
            "t_nodes": torch.tensor(0., device=device),
            "tt_edges": torch.tensor(0., device=device),
            "tPV_edges": torch.tensor(0., device=device),
            "ft_nodes": torch.tensor(0., device=device),
            "pv_asso": torch.tensor(0., device=device)}
    return loss



def get_block_score(log, weights, y, layer, var):
    sig_selbool = (y == 1).squeeze()
    weights = weights.to(torch.float16)
    log[f"sig_{var}_score_{layer}"] = torch.cat([weights[sig_selbool].cpu(), log[f"sig_{var}_score_{layer}"]])
    log[f"bkg_{var}_score_{layer}"] = torch.cat([weights[~sig_selbool].cpu(), log[f"bkg_{var}_score_{layer}"]])


def loss_logging(log, loss, configs, mode="DFEI"):
    if mode == "DFEI":
        if configs["node_prune"]:
            log["t_nodes_loss"].append(loss["t_nodes"].item())
        if configs["edge_prune"]:
            log["tt_edges_loss"].append(loss["tt_edges"].item())
        if configs["pv_asso"]:
            log["tpv_edges_loss"].append(loss["pv_asso"].item())
    elif mode == "IFT":
        if configs["frag"]:
            log["frag_loss"].append(loss["frag_nodes"].item())
        if configs["FT"]:
            log["ft_loss"].append(loss["ft_nodes"].item())
    return log


def binary_auc_score(y_true, y_score):
    y_true = torch.as_tensor(y_true, dtype=torch.int64).flatten()
    y_score = torch.as_tensor(y_score, dtype=torch.float32).flatten()
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"))

    order = torch.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = torch.empty_like(y_score)
    ranks[order] = torch.arange(1, y_score.numel() + 1, device=y_score.device, dtype=torch.float32)

    unique_scores, inverse, counts = torch.unique_consecutive(
        sorted_scores, return_inverse=True, return_counts=True
    )
    del unique_scores
    if torch.any(counts > 1):
        sorted_ranks = ranks[order]
        for group_idx, count in enumerate(counts):
            if count > 1:
                group_mask = inverse == group_idx
                sorted_ranks[group_mask] = sorted_ranks[group_mask].mean()
        ranks[order] = sorted_ranks

    rank_sum_pos = ranks[pos].sum()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ft_validation_metrics(logits, y_ft):
    selbool = y_ft != 1
    if selbool.sum() == 0:
        nan = torch.tensor(float("nan"), device=logits.device)
        return {"ft_auc": nan, "ft_tagging_power_proxy": nan, "ft_wrong_fraction_proxy": nan}

    probs = torch.softmax(logits[selbool], dim=1)
    labels = y_ft[selbool]
    y_true = (labels == 2).to(torch.int64)
    y_score = probs[:, 2]

    auc = binary_auc_score(y_true.detach().cpu(), y_score.detach().cpu()).to(logits.device)
    pred = torch.where(probs[:, 2] >= probs[:, 0], 2, 0)
    wrong_fraction = (pred != labels).to(torch.float32).mean()
    tagging_power_proxy = (1 - 2 * wrong_fraction) ** 2
    return {
        "ft_auc": auc,
        "ft_tagging_power_proxy": tagging_power_proxy * 100,
        "ft_wrong_fraction_proxy": wrong_fraction * 100,
    }


def epoch_end_loggable(log):

    avg_log = {}
    loss_keys = [k for k in log if k.endswith('_loss')]
    for key in loss_keys:
        avg_log[key] = torch.tensor(log[key]).nanmean(dim=0)

    metric_keys = [k for k in log if k in {"ft_auc", "ft_tagging_power_proxy", "ft_wrong_fraction_proxy"}]
    for key in metric_keys:
        avg_log[key] = torch.tensor(log[key], dtype=torch.float32).nanmean(dim=0)

    class_numbers = sorted(set(
        int(m.group(1)) for k in log if (m := re.search(r'LCA_class(\d+)_num', k))
    ))


    for score in class_numbers:
        selbool = torch.tensor(log[f"LCA_class{score}_num"]) != 0
        avg_log[f"LCA_class{score}_num"] = torch.tensor(log[f"LCA_class{score}_num"]).nanmean(dim=0)
        for s in class_numbers:
            avg_log[f"LCA_class{score}_pred_class{s}"] = torch.nan_to_num(
                torch.tensor(log[f"LCA_class{score}_pred_class{s}"])[selbool].nanmean(dim=0), nan=-1)

    return avg_log
