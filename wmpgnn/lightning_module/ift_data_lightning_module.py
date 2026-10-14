import pytorch_lightning as L

import copy
import warnings

import torch.nn as nn

from wmpgnn.lightning_module.lightning_helper import *
from wmpgnn.reconstruction_data.reconstruction import EventReconstruction


class IFTLightningModuleData(L.LightningModule):
    def __init__(self, model, dfei_model, configs):
        super().__init__()
        self.version = configs["settings"]["model"]

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


        if self.configs["frag"]:
            self.frag_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones(1))
        if self.configs["FT"]:
            self.ft_criterion = nn.CrossEntropyLoss(weight=torch.ones(3))

        self.dfei_model = dfei_model
        self.dfei_use_pid = configs['DFEI']["use_pid"] if "DFEI" in configs else "None"
        self.ift_use_pid = configs["IFT"]["use_pid"]

        if self.ift_use_pid == "true" or self.dfei_use_pid == "true":
            warnings.warn(
                "CONTAMINATION: use_pid='true' feeds true Monte-Carlo PID into the "
                "model. Oracle/upper-bound setting, not a realistic result.",
                stacklevel=2,
            )


        self.evt_reco = EventReconstruction(configs)


        self.edge_prune = configs["inference"]["edge_prune_thr"]
        self.node_prune = configs["inference"]["node_prune_thr"]

        self.log_dir = configs["log_dir"]

    def forward(self, batch):
        return self.model(batch)

    def test_step(self, batch, batch_idx):

        if self.dfei_model is not None:
            dfei_input = copy.deepcopy(batch)
            if self.dfei_use_pid == "realistic":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].real_pid], dim=1)
            elif self.dfei_use_pid == "true":
                dfei_input["tracks"].x = torch.cat([dfei_input["tracks"].x, dfei_input["tracks"].pid], dim=1)
            dfei_outputs = self.dfei_model(dfei_input)
            lca = dfei_outputs[("tracks", "to", "tracks")].edges
        elif "lca" in batch[("tracks", "to", "tracks")]:
            lca = batch[("tracks", "to", "tracks")].lca
        else:
            warnings.warn(
                "CONTAMINATION: no DFEI model and no cached LCA prediction; falling "
                "back to TRUE LCA labels (tracks.y) as IFT edge input. Oracle setting.",
                stacklevel=2,
            )
            lca = torch.nn.functional.one_hot(batch[("tracks", "tracks")].y.to(torch.long), num_classes=4).to(
                torch.float32)
        lca_score = torch.argmax(lca, dim=1).unsqueeze(1)
        batch[("tracks", "tracks")].edges = torch.cat([batch[("tracks", "tracks")].edges, lca_score], dim=1)


        if self.ift_use_pid == "realistic":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].real_pid], dim=1)
        elif self.ift_use_pid == "true":
            batch["tracks"].x = torch.cat([batch["tracks"].x, batch["tracks"].pid], dim=1)

        outputs_ft = self.model(batch)
        outputs_ft[("tracks", "to", "tracks")].lca = lca


        ft_des = torch.softmax(outputs_ft["tracks"].x, dim=1)



        if self.dfei_model is not None:
            outputs_ft["node_weights"] = self.dfei_model._blocks[-1].node_weights["tracks"].squeeze()
        elif "pred_y" in outputs_ft["tracks"]:
            outputs_ft["node_weights"] = outputs_ft["tracks"].pred_y
        else:
            raise ValueError("No node weights found")

        if self.dfei_model is not None:
            outputs_ft["edge_weights"] = self.dfei_model._blocks[-1].edge_weights[
                ('tracks', 'to', 'tracks')].squeeze()
        elif "pred_y" in outputs_ft[("tracks", "tracks")]:
            outputs_ft["edge_weights"] = outputs_ft[("tracks", "tracks")].pred_y
        else:
            raise ValueError("No edge weights found")

        self.evt_reco.reconstruct_heavyhadrons(outputs_ft, ft_des=ft_des)

        return {}

    def on_test_epoch_end(self):
        sig_df = self.evt_reco.collect_results()
        sig_df.to_csv(f'{self.log_dir}/IFT/version_{self.version}/signal_reco_data_df_{self.signal}.csv', index=False)
