import re
import numpy as np
import pandas as pd
import uproot

import torch
from pathlib import Path

from calculation_helper import get_derive_base_quant
from dtype_dict import dtype_dict
import zstandard as zstd


def save_compressed(data, filepath, level=3):
    cctx = zstd.ZstdCompressor(level=level)
    with open(filepath + '.zst', 'wb') as f:
        with cctx.stream_writer(f) as compressor:
            torch.save(data, compressor)

def get_simulation_type(dir_path: str):
    base_dir = Path(dir_path).name
    m = re.search(r"DFEI_([^_]+)_", base_dir)
    if m:
        return m.group(1)
    else:
        raise ValueError()


def get_decay(file: str):
    if file.split('_')[3] == 'inclusive':
        return file.split('_')[3]
    else:
        return file.split('_')[3] + '_' + file.split('_')[4]


def get_range(file: str):
    m = re.search(r"Event_data_particles_.*?_(\d+)_to_(\d+)\.root$", file)
    return int(m.group(1)), int(m.group(2))


def load_data(file: str, load_vars: list, decaytree: str) -> pd.DataFrame:
    with uproot.open(f"{file}") as f:
        particles = f[decaytree]
        df = particles.arrays(load_vars, library="pd")
    return df


def load_multiple_data(files: list[str], load_vars: list[str], decaytree: str, filters: str = None) -> pd.DataFrame:
    dfs = []
    for file in files:
        with uproot.open(f"{file}") as f:
            particles = f[decaytree]

            vars_to_load = load_vars.copy()
            if filters is not None:
                vars_to_load = vars_to_load + list(filters.keys())

            df = particles.arrays(vars_to_load, library="pd")

            if filters is not None:
                for col, target in filters.items():
                    df = df[df[col] == target]
                df = df.drop(columns=filters.keys())

        dfs.append(df)
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final


def get_keys(file: str, decaytree: str) -> list:
    with uproot.open(file) as f:
        tree = f[decaytree]
        return tree.keys()


def obtain_configs(configs: str = None,
                   edge_features=None, node_features=None, topo_classes=None, use_ft=False, edge_prune=False,
                   allBfinal=False, include_neutral=False) -> dict:
    if node_features is None:
        node_features = []
    if edge_features is None:
        edge_features = []
    if topo_classes is None:
        topo_classes = []

    if configs is not None:
        print("loading configs Need to be implemented")
        configs = {}
    else:
        configs = {"edge_features": edge_features, "node_features": node_features, "topo_classes": topo_classes,
                   "use_ft": use_ft,
                   "edge_prune": edge_prune, "allBfinal": allBfinal,
                   "include_neutral": include_neutral}

    print("=" * 15)
    print("Used configs:")
    for key in configs:
        print(f"{key}: {configs[key]}")
    return configs

def adjust_df_dtype(df):
    for key in df.keys():
        if key in dtype_dict.keys():
            df[key] = df[key].astype(dtype_dict[key])
    return df


def get_part_rel(part_file, rel_file, node_features, edge_features, configs=None, reco_suffix=''):
    if configs is None:
        configs = {}
    # Default particle information which needs to be read in
    default_part_info = ['ExpandedEventNumber', 'ExpandedParticleIndex', 'ExpandedPrimaryHeavyHadronIndex', 'id',
                         'ParticleType', f"associated_ip{reco_suffix}", f"pt{reco_suffix}",
                         'inGeomAcc', 'isCharged', 'isbhadron', 'isfromb', 'nFinals',
                         f'xPV_reco', f'yPV_reco', f'zPV_reco', f'xPV', f'yPV', f'zPV']
    default_rel_info = ["ExpandedEventNumber", "FirstParticleIndex", "SecondParticleIndex", "FromSameHeavyHadron",
                        "TopoLCA_ids", "TopoLCA_class0", "TopoLCA_class1", "TopoLCA_class2", "TopoLCA_class3",
                        "LCA",
                        "BothInGeomAcc", "p1_isCharged", "p2_isCharged"]

    additional_part = []
    additional_rel = []
    # additional relation variables necessary for early edge prune
    if configs['edge_prune']:
        additional_rel += [f"FromSameAssociatedPV{reco_suffix}", f"theta{reco_suffix}"]

    if configs['use_ft']:
        # afaik it is fragPartnerIndex is expanded
        additional_part += ["isfromb", "isFrag", 'fragType', 'fragPartnerIndex']

    if configs["sim"] == "LHCb":
        additional_part += ["EVENTNUMBER", "RUNNUMBER"]

    part_info = node_features + default_part_info + additional_part
    rel_info = edge_features + default_rel_info + additional_rel

    # check for variables which are not able to be load in and need to be derived later on
    part_keys = get_keys(part_file, "Particles")
    rel_keys = get_keys(rel_file, "ParticleRelations")
    to_derive = {'part': sorted(set(part_info) - set(part_keys)), 'rel': sorted(set(rel_info) - set(rel_keys))}
    part_base_quant, rel_base_quant = get_derive_base_quant(to_derive)
    part_loading = sorted(set(part_info + part_base_quant) & set(part_keys))
    rel_loading = sorted(set(rel_info+ rel_base_quant) & set(rel_keys))

    # Loading the data
    print("=" * 15)
    print("Loading in the data")
    part_data = load_data(part_file, part_loading, "Particles")
    rel_data = load_data(rel_file, rel_loading, "ParticleRelations")

    # adjust dtype
    part_data = adjust_df_dtype(part_data)
    rel_data = adjust_df_dtype(rel_data)

    return part_data, rel_data, to_derive


def filter_data(part_data, rel_data, configs=None, reco_suffix=''):
    # adding information to th edges for the edge selection
    part_data.set_index(['ExpandedEventNumber', 'ExpandedParticleIndex'], inplace=True)
    for idx, p in [('First', 'p1'), ('Second', 'p2')]:
        temp = part_data.reindex(pd.MultiIndex.from_arrays(
            [rel_data['ExpandedEventNumber'].values, rel_data[f'{idx}ParticleIndex'].values]))[
            [f"pt_reco", f"associated_ip{reco_suffix}"]]
        rel_data.loc[:, f'{p}_pt{reco_suffix}'] = temp[f'pt{reco_suffix}'].to_numpy(np.float32)
        rel_data.loc[:, f'{p}_associated_ip{reco_suffix}'] = temp[
            f'associated_ip{reco_suffix}'].to_numpy(np.float32)
    part_data.reset_index(inplace=True)
    del temp

    """Filter particle and relations based on acceptance"""
    # stable particles -> forming the nodes of the graph
    selbool = (part_data['ParticleType'] == 0) & (part_data['isCharged'] == 1) & (
            part_data[f'pt{reco_suffix}'] > 0) & (part_data[f'associated_ip{reco_suffix}'] >= 0)
    if configs['allBfinal']:
        geo_selbool = (part_data['inGeomAcc'] == 1) | (part_data["ExpandedPrimaryHeavyHadronIndex"] != -1)
    else:
        geo_selbool = (part_data['inGeomAcc'] == 1)
    selbool = selbool & geo_selbool
    stable_part = part_data[selbool].reset_index(drop=True)

    # All relations
    if configs['allBfinal']:
        is_B_final = allstable_part["ExpandedPrimaryHeavyHadronIndex"] != -1
        is_in_geom_acc = allstable_part['inGeomAcc'] == 1

        B_finalparticles = allstable_part.loc[is_B_final, "ExpandedParticleIndex"]
        stable_in_acc = allstable_part.loc[is_in_geom_acc, "ExpandedParticleIndex"]

        first_is_B = rel_data["FirstParticleIndex"].isin(B_finalparticles)
        second_is_B = rel_data["SecondParticleIndex"].isin(B_finalparticles)
        first_in_acc = rel_data["FirstParticleIndex"].isin(stable_in_acc)
        second_in_acc = rel_data["SecondParticleIndex"].isin(stable_in_acc)
        geo_selbool = (
                (rel_data['BothInGeomAcc'] == 1) |
                (first_is_B & second_in_acc) |
                (second_is_B & first_in_acc)
        )
    else:
        geo_selbool = rel_data['BothInGeomAcc'] == 1
    selbool = (rel_data['p1_isCharged'] == 1) & (rel_data['p2_isCharged'] == 1) & (
            rel_data[f'p1_pt{reco_suffix}'] > 0) & (rel_data[f'p2_pt{reco_suffix}'] > 0) & (
                      rel_data[f'p1_associated_ip{reco_suffix}'] >= 0) & (
                      rel_data[f'p2_associated_ip{reco_suffix}'] >= 0) & geo_selbool
    if configs['edge_prune']:
        edge_pre_selbool = (rel_data['FromSameAssociatedPV' + reco_suffix] == 1) | (
                rel_data['theta' + reco_suffix] <= 0.26)
        rel_data = rel_data[selbool & edge_pre_selbool]
    else:
        rel_data = rel_data[selbool]

    # Signal relations
    selbool = ((rel_data['FromSameHeavyHadron'] == 1) &
               (rel_data['p1_isCharged'] == 1) & (rel_data['p2_isCharged'] == 1))
    if not configs['allBfinal']:
        geo_selbool = rel_data['BothInGeomAcc'] == 1
        selbool = selbool & geo_selbool
    allsig_rel = rel_data[selbool].copy().reset_index(drop=True)
    print("=" * 15)

    # b particles within the event
    print('Treating only B as signal')
    data_b_particles = part_data[(part_data['isbhadron'] == 1)]
    del part_data

    return stable_part, rel_data, data_b_particles, allsig_rel
