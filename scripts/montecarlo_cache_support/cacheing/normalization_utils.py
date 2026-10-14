import numpy as np
import torch
import os
import glob

from helper import *
from calculation_helper import derive_quantities, get_derive_base_quant

def build_norm_df(df):
    stats = pd.DataFrame({
        "skew": df.skew().astype("float32"),
        "mean": df.mean().astype("float32"),
        "std": df.std().astype("float32"),
        "median": df.median().astype("float32"),
        "iqr": (df.quantile(0.25) - df.quantile(0.75)).astype("float32")
    }).T


    bounded = (df.min() >= -2) & (df.max() <= 2)
    bounded_cols = bounded[bounded].index


    stats.loc["mean", bounded_cols] = np.float32(0)
    stats.loc["median", bounded_cols] = np.float32(0)
    stats.loc["std", bounded_cols] = np.float32(1)
    stats.loc["iqr", bounded_cols] = np.float32(1)

    return stats

def obtain_normalization(file, configs, force_recalculation=False):
    part_file = os.path.basename(file)
    rel_file = part_file.replace("particles", "relations")


    m = re.search(r'_inclusive_([^_]+)_\d{8}_to_\d{8}\.root', part_file)
    condition = m.group(1) if m else None
    if condition is None: raise NotImplementedError


    if os.path.exists("normalization_dict.pt"):
        normalization_dict = torch.load("normalization_dict.pt", weights_only=False)
        if condition in normalization_dict.keys():
            keys = list(normalization_dict[condition].keys())
        else:
            normalization_dict[condition] = {}
            keys = []
        if force_recalculation:
            node_features = configs["node_features"]
            edge_features = configs["edge_features"]
        else:
            node_features = sorted(set(configs["node_features"]) - set(keys))
            edge_features = sorted(set(configs["edge_features"]) - set(keys))

        if not node_features and not edge_features:
            return normalization_dict[condition]
    else:
        normalization_dict = {condition: {}}
        node_features = configs["node_features"]
        edge_features = configs["edge_features"]

    """Here we need to read in inclusive decay files to obtain normalization information"""

    indir = os.path.dirname(os.path.dirname(file))
    inclusive_dirs = glob.glob(f"{indir}/*inclusive")
    if not inclusive_dirs:
        missing = sorted(set(node_features + edge_features))
        raise FileNotFoundError(
            f"Missing inclusive normalization source for condition {condition}. "
            f"Could not find {indir}/*inclusive and cached stats lack: {missing}"
        )
    indir = inclusive_dirs[0]

    print("Obtaining normalization information")
    part_keys = get_keys(f"{indir}/{part_file}", "Particles")
    rel_keys = get_keys(f"{indir}/{rel_file}", "ParticleRelations")


    to_derive = {'part': sorted(set(node_features) - set(part_keys)),
                 'rel': sorted(set(edge_features) - set(rel_keys))}
    part_base_quant, rel_base_quant = get_derive_base_quant(to_derive)
    part_loading = sorted(set(node_features) & set(part_keys)) + part_base_quant
    rel_loading = sorted(set(edge_features) & set(rel_keys)) + rel_base_quant

    m = re.search(fr'^.*?{condition}', part_file)
    grab_file = m.group(0) if m else None
    if grab_file is None: raise NotImplementedError

    particle_files = sorted(glob.glob(f'{indir}/{grab_file}*'))[:10]
    relation_files = sorted(glob.glob(f'{indir}/{grab_file.replace("particles", "relations")}*'))[:10]


    part_filter = {"isFinal": 1, "isCharged": 1, "inGeomAcc": 1}
    part_df = load_multiple_data(particle_files, part_loading, "Particles", filters=part_filter)
    part_df[part_df.select_dtypes("bool").columns] = part_df.select_dtypes("bool").astype(int)

    rel_filter = {"BothInGeomAcc": 1, "p1_isCharged": 1, "p2_isCharged": 1}
    rel_df = load_multiple_data(relation_files, rel_loading, "ParticleRelations", filters=rel_filter)
    rel_df[rel_df.select_dtypes("bool").columns] = rel_df.select_dtypes("bool").astype(int)


    part_df, rel_df = derive_quantities(part_df, rel_df, to_derive)
    part_norm = build_norm_df(part_df)
    rel_norm = build_norm_df(rel_df)


    new_normalization = pd.concat([rel_norm, part_norm], axis=1).to_dict()

    normalization_dict[condition].update(new_normalization)
    torch.save(normalization_dict, "normalization_dict.pt")

    return normalization_dict[condition]


def adjust_normalization(normalization_dict, configs, mode: str = "auto"):
    normalization = pd.DataFrame(normalization_dict).T.loc[configs["edge_features"] + configs["node_features"]]

    if mode == "auto":
        selected = pd.DataFrame(index=normalization.index, columns=['value'])
        skew_thresh = 1.0
        for feature in normalization.index:
            if abs(normalization.loc[feature, 'skew']) > skew_thresh:
                selected.loc[feature, 'value'] = (normalization.loc[feature, 'median'],
                                                  normalization.loc[feature, 'iqr'])
            else:
                selected.loc[feature, 'value'] = (normalization.loc[feature, 'mean'],
                                                  normalization.loc[feature, 'std'])

        selected[['val1', 'val2']] = pd.DataFrame(selected['value'].tolist(), index=selected.index)
        selected = selected.drop(columns='value')
    else:
        raise NotImplementedError("Implement only selecting mean or median")

    selected[selected["val2"] == 0] = 1
    return selected
