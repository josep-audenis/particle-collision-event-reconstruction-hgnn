












import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

import torch

import os
import copy
import re

import pandas as pd

import yaml
from optparse import OptionParser

from wmpgnn.analysis.trainer_helper import *
from wmpgnn.data_loader.data_loader_helper import load_tst_loader
from wmpgnn.analysis.weights_calculator import transform_pos_weight
from wmpgnn.performance.plotter import metrics_eval
from wmpgnn.lightning_module.dfei_lightning_module import DFEILightningModule
from wmpgnn.lightning_module.exec_lightning import load_module, training, evaluate
from wmpgnn.util.pruners import true_node_pruning


def check_data(evt):
    for key in evt["tracks"].keys():
        if evt["tracks"][key].shape[0] == 0:
            print(key)
            return False
    for key in evt["pvs"].keys():
        if evt["pvs"][key].shape[0] == 0:
            print(key)
            return False
    for key in evt[("tracks", "to", "tracks")].keys():
        if evt[("tracks", "to", "tracks")][key].shape[0] == 0:
            print(key)
            return False
    for key in evt[("tracks", "to", "pvs")].keys():
        if evt[("tracks", "to", "pvs")][key].shape[0] == 0:
            print(key)
            return False
    return True


class ChunkMonitorCallback(Callback):
    def __init__(self):
        self.file_monitor = 0

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch["last_chunk"]:
            outdir = pl_module.configs["evaluate"]["pv_output_dir"] + "/" + pl_module.configs["evaluate"]["sample"]
            os.makedirs(outdir, exist_ok=True)
            file = f"{outdir}/{pl_module.configs["evaluate"]["data_set"]}_data_{self.file_monitor}.pt"
            self.file_monitor += 1
            torch.save(pl_module.pv_asso_holder, file)
            pl_module.pv_asso_holder = []


class pv_asso_module(L.LightningModule):

    def __init__(self, model, configs):
        super().__init__()
        self.model = model
        self.configs = configs
        self.pv_asso_holder = []


    def forward(self, batch):
        return self.model(batch)

    def test_step(self, batch, batch_idx):
        original_data = copy.deepcopy(batch)

        outputs = self.model(batch)
        original_data[('tracks', 'to', 'tracks')].lca = outputs[('tracks', 'to', 'tracks')].edges
        original_data["tracks"].pred_y = self.model._blocks[-1].node_weights["tracks"].squeeze()
        original_data[('tracks', 'to', 'tracks')].pred_y = self.model._blocks[-1].edge_weights[
            ('tracks', 'to', 'tracks')].squeeze()


        ntracks = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][0]).shape[0]
        npvs = torch.unique(outputs[("tracks", "to", "pvs")]["edge_index"][1]).shape[0]
        y_pv = torch.argmax(outputs[("tracks", "to", "pvs")].y.view(ntracks, npvs), dim=1)
        pred_pv = torch.argmax(self.model._blocks[-1].edge_weights[('tracks', 'to', 'pvs')].view(ntracks, npvs), dim=1)


        node_selbool = outputs["tracks"].ft != 1
        pv_oi = torch.unique(pred_pv[node_selbool])
        for pv in pv_oi:
            pv_oi_data = copy.deepcopy(original_data)
            nodes_asso_pv_selbool = pred_pv == pv
            pv_selbool = torch.zeros(pv_oi_data["pvs"].x.shape[0], dtype=torch.bool, device=self.device)
            pv_selbool[pv] = True


            true_node_pruning(nodes_asso_pv_selbool, pv_oi_data,
                              "tracks", [('tracks', 'to', 'tracks'), ('tracks', 'to', 'pvs')])


            true_node_pruning(pv_selbool, pv_oi_data, "pvs", [('tracks', 'to', 'pvs')])
            del pv_oi_data["last_chunk"]
            isfine = check_data(pv_oi_data)
            if isfine:
                self.pv_asso_holder.append(pv_oi_data.to('cpu'))
            else:
                raise RuntimeError("Invalid PV-associated graph")
        return 0


if __name__ == "__main__":

    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))


    with open(option.CONFIG, "r") as file:
        configs = yaml.safe_load(file)

    model = "None"
    if "IFT" in option.CONFIG:
        model = "IFT"
    elif "DFEI" in option.CONFIG:
        model = "DFEI"
    else:
        model = configs["evaluate"]["model_arch"]

        hparams_file = f"lightning_logs/{model}/version_{configs["evaluate"]["model"]}/hparams.yaml"
        with open(hparams_file, "r") as file:
            hparams = yaml.safe_load(file)
        configs[model] = hparams[model]

        configs[model]["settings"]["data_dir"] = configs["evaluate"]["data_dir"]
        if "nu" not in configs[model]["settings"]["data_dir"]:
            raise ValueError("multiplicity within the data is required for reassociation")
        configs[model]["settings"]["ncpu"] = configs["evaluate"]["ncpu"]
    print(f"Evaluation script started of {model}")
    print("=" * 30)


    configs[model]["data_overwrite"] = configs["evaluate"]["data_set"]
    configs, tst_loader, chunkloader = load_tst_loader(configs, model=model)


    pos_weights = transform_pos_weight(None, None, mode="eval")
    print("DFEI module:")
    configs[model]["cpt"] = configs["evaluate"]["model"]
    module = load_module(configs, pos_weights, model="DFEI", is_train=False)
    dfei_model = module.model

    trainer = Trainer(
        callbacks=[ChunkMonitorCallback()],
        default_root_dir=f'lightning_logs/version_0',
    )

    if tst_loader is not None:
        trainer.test(pv_asso_module(dfei_model, configs), dataloaders=tst_loader)
    else:
        trainer.test(pv_asso_module(dfei_model, configs), dataloaders=chunkloader)
