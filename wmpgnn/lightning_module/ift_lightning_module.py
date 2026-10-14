import pytorch_lightning as L

from collections import defaultdict
import copy
import warnings

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.util.pruners import edge_pruning, true_node_pruning
from wmpgnn.reconstruction.reconstruction import EventReconstruction
from wmpgnn.performance.reco_accuracy import obtain_reco_accuracy
from wmpgnn.performance.plotter import *
from wmpgnn.performance.tagging_power import analyze_tagging_power
from wmpgnn.performance.plot_results import process_ft


class IFTLightningModule(L.LightningModule):
    def __init__(self, model, dfei_model, optimizer_class, optimizer_params, configs, pos_weights):
        super().__init__()
        if "model" in configs["settings"]:
            self.version = configs["settings"]["model"]
        else:
            self.version = None
            self.save_hyperparameters({
                **configs,
                "pos_weights": make_loggable(pos_weights)
            })

        self.signal = "_".join(configs["evaluate"]["sample"])
        if configs["evaluate"]["over_write"] != "":
            self.signal += "__" + configs["evaluate"]["over_write"]
        _split = configs["evaluate"].get("split", "tst")
        if _split != "tst":
            # Tag non-test evaluations so val-selection outputs never overwrite or
            # get mistaken for the final test-set result.
            self.signal += "__" + _split

        self.configs = configs["inference"]
        self.model = model

        self.dfei_model = [dfei_model]
        if self.dfei_model[0] is not None:
            for param in self.dfei_model[0].parameters():
                param.requires_grad = False
        self.dfei_use_pid = configs['DFEI']["use_pid"] if "DFEI" in configs else "None"
        self.ift_use_pid = configs["IFT"]["use_pid"]
        self.lca_input = configs["IFT"].get("lca_input", "argmax")
        # FT loss variant (config/env knob IFT_LOSS): "ce" = current weighted
        # cross-entropy on b/bbar tag tracks only; "allclass" = weighted CE over all
        # 3 classes incl. the "none" sea; "focal" = focal loss on the tag tracks.
        # "dsquared" (per-event dilution^2 surrogate) is intentionally not yet enabled
        # until validated, so a suite entry can't silently train a wrong objective.
        self.loss_type = configs["IFT"].get("loss", "ce")
        self.focal_gamma = float(configs["IFT"].get("focal_gamma", 2.0))
        if self.loss_type == "dsquared":
            raise NotImplementedError(
                "IFT loss 'dsquared' (per-event D^2 surrogate) is not yet implemented/"
                "validated. Use ce|allclass|focal."
            )

        if self.ift_use_pid == "true" or self.dfei_use_pid == "true":
            warnings.warn(
                "CONTAMINATION: use_pid='true' feeds true Monte-Carlo PID into the "
                "model. This is an oracle/upper-bound setting, not a realistic result. "
                "Use use_pid='realistic' for the headline number.",
                stacklevel=2,
            )

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        if self.configs["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights["frag"])
        if self.configs["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=pos_weights["FT"])

        self.trn_log, self.val_log = init_logs(configs)
        self.tst_log = init_logs(configs, mode="test")

        self.evt_reco = EventReconstruction(configs)


        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_params)

    def shared_step(self, batch, batch_idx, log, mode="train"):
        optimizers = self.optimizers()
        loss = init_loss(self.device)


        if self.dfei_model[0] is not None:
            dfei_input = copy.deepcopy(batch)
            if self.dfei_use_pid == "realistic":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].real_pid], dim=1)
            elif self.dfei_use_pid == "true":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].pid], dim=1)
            self.dfei_model[0] = self.dfei_model[0].to(self.device)
            dfei_outputs = self.dfei_model[0](dfei_input)
            lca = dfei_outputs[("tracks", "to", "tracks")].edges
        elif "lca" in batch[("tracks", "to", "tracks")]:
            lca = batch[("tracks", "to", "tracks")].lca
        else:
            warnings.warn(
                "CONTAMINATION: no DFEI model and no cached LCA prediction; falling "
                "back to TRUE LCA labels (tracks.y) as IFT edge input. This is an "
                "oracle setting, not a realistic result.",
                stacklevel=2,
            )
            lca = torch.nn.functional.one_hot(batch[("tracks", "tracks")].y.to(torch.long), num_classes=4).to(
                torch.float32)
        if self.lca_input == "argmax":
            lca_features = torch.argmax(lca, dim=1).unsqueeze(1).to(batch[("tracks", "tracks")].edges.dtype)
        elif self.lca_input == "probabilities":
            lca_features = torch.softmax(lca, dim=1)
        else:
            raise ValueError(f"Unknown IFT lca_input: {self.lca_input}")
        lca_score = torch.argmax(lca, dim=1).unsqueeze(1)
        batch[("tracks", "tracks")].edges = torch.cat([batch[("tracks", "tracks")].edges, lca_features], dim=1)


        if self.ift_use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.ift_use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)


        outputs_ft = self.model(batch)
        outputs_ft[("tracks", "to", "tracks")].lca = lca

        y_ft = batch['tracks'].ft
        if self.loss_type == "allclass":
            # supervise every track incl. the "none" class (no selbool drop)
            logits, target = outputs_ft["tracks"].x, y_ft
        else:
            selbool = y_ft != 1
            logits, target = outputs_ft["tracks"].x[selbool], y_ft[selbool]
        if self.loss_type == "focal":
            ce = nn.functional.cross_entropy(
                logits, target, weight=self.ft_criterion.weight, reduction="none")
            pt = torch.exp(-ce)
            ift_loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()
        else:  # "ce" or "allclass"
            ift_loss = self.ft_criterion(logits, target)
        loss["ft_nodes"] += ift_loss
        if mode == "val":
            val_metrics = ft_validation_metrics(outputs_ft["tracks"].x, y_ft)
            for key, value in val_metrics.items():
                log[key].append(value.detach().cpu().item())


        if mode == "test":
            ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)
            if "frag" in outputs_ft["tracks"]:
                frag_selbool = outputs_ft["tracks"].frag != -1
                frag_in_evt = outputs_ft["tracks"].frag[frag_selbool]
                frag_pid = outputs_ft["part_ids"][frag_selbool]



            if self.dfei_model[0] is not None:
                outputs_ft["node_weights"] = self.dfei_model[0]._blocks[-1].node_weights["tracks"].squeeze()
            elif "pred_y" in outputs_ft["tracks"]:
                outputs_ft["node_weights"] = outputs_ft["tracks"].pred_y
            else:
                warnings.warn(
                    "CONTAMINATION: no DFEI model and no predicted node weights; using "
                    "TRUE node labels (tracks.ft) as reco node weights. Oracle setting.",
                    stacklevel=2,
                )
                outputs_ft["node_weights"] = (outputs_ft["tracks"].ft != 1).int()

            if self.dfei_model[0] is not None:
                outputs_ft["edge_weights"] = self.dfei_model[0]._blocks[-1].edge_weights[
                    ('tracks', 'to', 'tracks')].squeeze()
            elif "pred_y" in outputs_ft[("tracks", "tracks")]:
                outputs_ft["edge_weights"] = outputs_ft[("tracks", "tracks")].pred_y
            else:
                warnings.warn(
                    "CONTAMINATION: no DFEI model and no predicted edge weights; deriving "
                    "reco edge weights from the LCA input (truth-derived if no DFEI). "
                    "Oracle setting.",
                    stacklevel=2,
                )
                outputs_ft["edge_weights"] = (lca_score.squeeze() != 0).int()

            if "frag" in outputs_ft["tracks"]:
                outputs_ft["frag_y"] = frag_in_evt
                outputs_ft["frag_pid"] = frag_pid
            self.evt_reco.reconstruct_heavyhadrons(outputs_ft, ft_des=ft_des)

        log = loss_logging(log, loss, self.configs, mode="IFT")
        return ift_loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.trn_log, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, self.val_log, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, batch_idx, self.tst_log, mode="test")
        return {}

    def on_train_epoch_end(self):
        avg_losses = epoch_end_loggable(self.trn_log)
        for key, val in avg_losses.items():
            self.log(f"train_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.trn_log = defaultdict(list)

    def on_validation_epoch_end(self):
        avg_losses = epoch_end_loggable(self.val_log)
        for key, val in avg_losses.items():
            self.log(f"val_{key}", val, prog_bar=(key == "ft_loss"), on_epoch=True, on_step=False)
        self.val_log = defaultdict(list)

    def on_test_epoch_end(self):
        if self.version is None:
            self.version = self.logger.version
        sig_df, evt_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/signal_reco_df_{self.signal}.csv', index=False)
        evt_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/event_reco_df_{self.signal}.csv', index=False)
        obtain_reco_accuracy(sig_df, self.version, self.signal, self.log_dir, model="IFT")
        """# Removing heavy hadron daughters of B since they are classified as signal (Ds in Bs->Dspi for example)
        if "Bs" in self.signal:
            sig_id = 531
        elif "Bd" in self.signal:
            sig_id = 511
        else:
            ValueError("Currently undefined")
        sig_selbool = sig_df["SigMatch"] == 1
        sig_id_selbool = np.abs(sig_df["B_id"]) != sig_id
        sig_df = sig_df[~(sig_selbool * sig_id_selbool)]"""
        if self.configs["FT"]:
            process_ft(self.tst_log, sig_df, self.version, self.signal, log_dir=self.log_dir)
            analyze_tagging_power(sig_df, self.version, self.signal, log_dir=self.log_dir)

            if "is_whiten" in sig_df.keys():
                whiten = self.signal + "__is_whiten"
                obtain_reco_accuracy(sig_df[sig_df["is_whiten"] == 1], self.version, whiten, self.log_dir, model="IFT")
                whiten_df = sig_df[sig_df["is_whiten"] == 1]
                process_ft(self.tst_log, whiten_df, self.version, whiten, log_dir=self.log_dir)
                analyze_tagging_power(whiten_df, self.version, whiten, log_dir=self.log_dir)
