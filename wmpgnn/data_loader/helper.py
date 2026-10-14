import copy
import io
import zstandard as zstd
from itertools import chain

import torch
from torch_geometric.loader import DataLoader

from wmpgnn.util.pruners import *
from wmpgnn.util.pv_association import pv_associate_data
from wmpgnn.calibration.calibration_mask import *
from wmpgnn.data_loader.weights_calculator import get_hetero_weight


def _feature_index(store, feature_name):
    """Locate a node feature column by name. Raises rather than guessing an
    index, since a wrong fallback index silently masks the wrong feature and
    leaks signal information."""
    names = getattr(store, "feature_names", None)
    if names is None:
        raise RuntimeError(
            "Signal input masking requires tracks.feature_names in cached graphs "
            "to locate the feature column safely. Rebuild the cache with updated "
            "GraphConstruction.py (no hardcoded-index fallback is used, since a "
            "wrong index would silently mask the wrong feature)."
        )
    if feature_name not in names:
        raise RuntimeError(
            f"Feature '{feature_name}' not found in tracks.feature_names {list(names)}."
        )
    return names.index(feature_name)


# Kinematic features that still reveal the signal-B decay topology even after
# charge/PID are masked. Zeroing them is required for a leakage-free opposite-side
# tag (enable via settings.mask_signal_kinematics).
_SIGNAL_KINEMATIC_FEATURES = (
    "px_reco", "py_reco", "pz_reco",
    "xProd_reco", "yProd_reco", "zProd_reco",
)


def mask_signal_track_inputs(data, mask_kinematics=False):
    if not hasattr(data["tracks"], "asso_hh"):
        raise RuntimeError(
            "IFT signal input masking requires tracks.asso_hh in cached graphs. "
            "Rebuild the cache with updated GraphConstruction.py."
        )

    signal_mask = data["tracks"].asso_hh > 0

    if not torch.any(signal_mask):
        return data

    charge_idx = _feature_index(data["tracks"], "charge")
    data["tracks"].x[signal_mask, charge_idx] = 0

    if mask_kinematics:
        names = data["tracks"].feature_names
        for feat in _SIGNAL_KINEMATIC_FEATURES:
            if feat in names:
                data["tracks"].x[signal_mask, names.index(feat)] = 0

    if hasattr(data["tracks"], "pid") and data["tracks"].pid is not None:
        data["tracks"].pid[signal_mask] = 0

    if hasattr(data["tracks"], "real_pid") and data["tracks"].real_pid is not None:
        data["tracks"].real_pid[signal_mask] = 0

    return data


def should_mask_signal_inputs(configs):
    if configs.get("model") != "IFT":
        return False
    return configs["settings"].get("mask_signal_inputs", True)


def should_exclude_signal_tracks(configs):
    if configs.get("model") != "IFT":
        return False
    return configs["settings"].get("exclude_signal_tracks", False)


def exclude_signal_tracks(data):
    if not hasattr(data["tracks"], "asso_hh"):
        raise RuntimeError(
            "IFT signal-track exclusion requires tracks.asso_hh in cached graphs. "
            "Rebuild the cache with updated GraphConstruction.py."
        )
    keep_mask = data["tracks"].asso_hh <= 0
    if not torch.any(keep_mask):
        return None
    true_node_pruning(keep_mask, data, "tracks", [("tracks", "to", "tracks"), ("tracks", "to", "pvs")])
    if data[("tracks", "to", "tracks")].edge_index.shape[1] == 0:
        return None
    return data


def get_nfiles(_configs):
    samples = _configs["sample"]
    nfiles = {}
    for sample, nfile in zip(samples, _configs["nfiles"]):
        nfiles[sample] = nfile
    return nfiles


def load_dataset(path, configs, mode="train", pv_asso_model=None):
    dctx = zstd.ZstdDecompressor()
    with open(path, 'rb') as f:
        with dctx.stream_reader(f) as reader:

            decompressed = reader.read()

            data = torch.load(io.BytesIO(decompressed), weights_only=False)

    """Applying pruning for different using truth pruning initially"""
    if "true" in configs["settings"]["graph_mode"]:
        data_selbool = torch.ones(len(data))
        edge_types = [("tracks", "to", "tracks"), ("tracks", "to", "pvs")]
        for i, evt in enumerate(data):
            y_nodes = evt["tracks"].ft != 1
            if "frag" in configs["settings"]["graph_mode"]:
                frag_selbool = evt["tracks"].frag != 0
                y_nodes = y_nodes | frag_selbool
            if configs["settings"]["node_sel"] == "true":
                true_node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "default":
                node_pruning(y_nodes, evt, "tracks", edge_types)
            elif configs["settings"]["node_sel"] == "zeros":
                test_node_pruning(y_nodes, evt, "tracks", edge_types)
            if evt[("tracks", "to", "tracks")].y.shape[0] == 0 or torch.all(evt[("tracks", "to", "tracks")].y == 0):
                data_selbool[i] = 0
        filtered_data = [d for d, sel in zip(data, data_selbool) if sel]
    else:
        filtered_data = data

    """Making the graph bidirectional"""
    for data in filtered_data:
        edge_type = ('tracks', 'to', 'tracks')
        store = data[edge_type]

        store.edge_index = torch.cat([store.edge_index, store.edge_index.flip(0)], dim=1)
        store.edges = store.edges.repeat(2, 1)
        if hasattr(store, 'y') and store.y is not None:
            store.y = store.y.repeat(2)

        if hasattr(store, 'lca') and store.lca is not None:
            store.lca = store.lca.repeat(2, 1)
        if hasattr(store, 'pred_y') and store.lca is not None:
            store.pred_y = store.pred_y.repeat(2)

    """PV asso"""
    if pv_asso_model is not None:
        ncpus = int(configs["settings"]["ncpu"] / 2)
        pv_data = DataLoader(filtered_data, batch_size=200)
        filtered_data = []
        for evt in pv_data:
            if pv_asso_model.name == "pv_asso_module":
                original_data = copy.deepcopy(evt)
                metrics = pv_asso_model.forward(evt)
                res = pv_associate_data(original_data, metrics, node_thr=pv_asso_model.node_thrs, n_cores=ncpus)
            else:
                metrics = pv_asso_model.forward(evt)
                res = pv_associate_data(evt, metrics, n_cores=ncpus)
            filtered_data.append(res)
        filtered_data = list(chain.from_iterable(filtered_data))

    """Whitening for calibration"""
    if configs["settings"]["calibration"]:
        filtered_data = adjust_for_calibration(configs, path, filtered_data, n_cores=int(configs["settings"]["ncpu"] / 2))

    if should_exclude_signal_tracks(configs):
        filtered_data = [evt for evt in (exclude_signal_tracks(evt) for evt in filtered_data) if evt is not None]

    if should_mask_signal_inputs(configs):
        mask_kin = configs["settings"].get("mask_signal_kinematics", False)
        filtered_data = [mask_signal_track_inputs(evt, mask_kinematics=mask_kin) for evt in filtered_data]

    if mode == "weights_only":
        weights = get_hetero_weight(filtered_data, configs)
        return weights
    elif "weights" in mode:
        weights = get_hetero_weight(filtered_data, configs)
        return filtered_data, weights
    else:
        return filtered_data
