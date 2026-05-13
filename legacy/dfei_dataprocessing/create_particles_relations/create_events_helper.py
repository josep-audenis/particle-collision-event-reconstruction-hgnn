from typing import Tuple

import numpy as np
import pandas as pd

from compute_mothers import compute_lists_of_mothers, compute_mother_PID
from calculation_helper import *


def get_evt_df(col_idx_list: list, col_offset_map: dict, event: int, *dfs: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Extract dataframes from multiple groups for given collision indices."""
    res = []
    for df in dfs:
        group = df.groupby("EventNumber")
        valid_evts = [evt for evt in col_idx_list if evt in group.groups]

        if not valid_evts:
            res.append(pd.DataFrame())
            continue

        if len(valid_evts) == 1:
            result_df = group.get_group(valid_evts[0]).copy()
        else:
            result_df = pd.concat(
                [group.get_group(evt) for evt in valid_evts],
                ignore_index=True
            )

        if not result_df.empty:
            result_df['ExpandedEventNumber'] = event
            PV_offset = result_df["EventNumber"].map(col_offset_map)

            # Direct addition for always-valid indices
            for key in ["ParticleIndex", "MotherIndex"]:
                result_df[f"Expanded{key}"] = result_df[key] + PV_offset

            # Conditional addition for nullable indices
            for key in ["fragPartnerIndex", "PrimaryHeavyHadronIndex"]:
                result_df[f"Expanded{key}"] = np.where(
                    result_df[key] != -1,
                    result_df[key] + PV_offset,
                    result_df[key]
                )

        res.append(result_df)
    return tuple(res)


def get_evt_df_fullsim(event: int, *dfs: pd.DataFrame):
    # optimized for LHCb sim as they are usually smaller files and don't need much processing
    res = []
    for df in dfs:
        result_df = df.loc[df["EventNumber"] == event].copy()
        # Just copying the normal to expanded so i dont have to change it downstream
        result_df['ExpandedEventNumber'] = result_df["EventNumber"]
        for key in ["ParticleIndex", "MotherIndex"]:
            result_df[f"Expanded{key}"] = result_df[key]
        # In principal one can also add here the frag one but they dont exist in full sim yet
        for key in ["PrimaryHeavyHadronIndex"]: # fragPartnerIndex
            result_df[f"Expanded{key}"] = np.where(
                result_df[key] != -1,
                result_df[key],
                result_df[key]
            )
        res.append(result_df)

    combined = pd.concat(res, ignore_index=True)
    true_pvs, idx = np.unique(combined[["xPV", "yPV", "zPV"]], axis=0, return_index=True)
    reco_pvs, idx = np.unique(combined[["xPV_reco", "yPV_reco", "zPV_reco"]], axis=0, return_index=True)

    # intermediate dont have any pv but default of -1, which are removed here
    true_pvs = true_pvs[~np.all(true_pvs == -1, axis=1)]
    reco_pvs = reco_pvs[~np.all(reco_pvs == -1, axis=1)]
    return true_pvs, reco_pvs, tuple(res)


def process_heavy_hadrons(heavy_hadron_df: pd.DataFrame) -> pd.DataFrame:
    particle_df = heavy_hadron_df.copy()
    particle_df['ParticleType'] = 2
    particle_df['FromPrimaryHeavyHadron'] = 0

    particle_df['ExpandedPrimaryHeavyHadronIndex'] = -1
    particle_df['MotherId'] = np.array(0, dtype=np.int32)

    pv_ip_cols = ["associated_xPV", "associated_yPV", "associated_zPV", "associated_ip",
                  "associated_xPV_reco", "associated_yPV_reco", "associated_zPV_reco", "associated_ip_reco"]
    particle_df[pv_ip_cols] = -1.

    for col in ['MotherList', 'TopoMotherList', 'MotherList_ids', 'TopoMotherList_ids']:
        particle_df[col] = [[] for _ in range(len(particle_df))]

    particle_df['pt'] = np.sqrt(particle_df['px'] ** 2 + particle_df['py'] ** 2)
    particle_df['pt_reco'] = np.sqrt(particle_df['px_reco'] ** 2 + particle_df['py_reco'] ** 2)
    particle_df['isfromb'] = 0
    return particle_df


def process_finals(stable_charged_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices,
                   true_pvs, reco_pvs, decay_dict):
    stable_charged_df['ParticleType'] = 0

    n_stable_charged = len(stable_charged_df)
    for key in ["MotherList", "TopoMotherList", "MotherList_ids", "TopoMotherList_ids"]:
        stable_charged_df[key] = [[]] * n_stable_charged

    decay_dict = compute_lists_of_mothers(stable_charged_df, unstable_from_hh_df, heavy_hadron_df, heavy_hadron_indices,
                                          decay_dict)
    if decay_dict == 0:
        evt_number = np.unique(stable_charged_df['EventNumber'])
        print('Mother obtaining failed at event ', evt_number)
        return 0
    # here we directly adjust the stable and need to adjust the heavy hadron indices
    # adjust stable charged
    # promote unstable_hh to hh_df -> Shouldnt be used anymore if it is merged
    # change hh indicies but they arent used anymore if not merged
    # If from a topological perspective a particle does not have a mother. remove them from ExpandedPrimaryHeavyHadronIndex
    for particle, head in decay_dict["topo_ancestors_per_particle"].items():
        if not head:
            stable_charged_df.loc[
                stable_charged_df['ExpandedParticleIndex'] == particle,
                'ExpandedPrimaryHeavyHadronIndex'
            ] = -1

    # stable_charged_df[stable_charged_df["ExpandedPrimaryHeavyHadronIndex"] != -1]
    stable_charged_df = compute_mother_PID(stable_charged_df, unstable_from_hh_df, heavy_hadron_df,
                                           heavy_hadron_indices)
    stable_charged_df = compute_pv_asso(stable_charged_df, true_pvs, "true")
    stable_charged_df = compute_pv_asso(stable_charged_df, reco_pvs, "reco")

    stable_charged_df["pt"] = np.sqrt(stable_charged_df['px'] ** 2 + stable_charged_df['py'] ** 2)
    stable_charged_df["pt_reco"] = np.sqrt(stable_charged_df['px_reco'] ** 2 + stable_charged_df['py_reco'] ** 2)

    from_primary_selbool = stable_charged_df['PrimaryHeavyHadronIndex'].isin(heavy_hadron_indices)
    stable_charged_df['FromPrimaryHeavyHadron'] = from_primary_selbool.astype(int)

    stable_charged_df.loc[from_primary_selbool, 'MotherList'] = stable_charged_df.loc[
        from_primary_selbool, 'ExpandedParticleIndex'].map(decay_dict["ancestors_per_particle"])
    stable_charged_df.loc[from_primary_selbool, 'TopoMotherList'] = stable_charged_df.loc[
        from_primary_selbool, 'ExpandedParticleIndex'].map(decay_dict["topo_ancestors_per_particle"])
    stable_charged_df.loc[from_primary_selbool, 'MotherList_ids'] = stable_charged_df.loc[
        from_primary_selbool, 'ExpandedParticleIndex'].map(decay_dict["ancestors_ids_per_particle"])
    stable_charged_df.loc[from_primary_selbool, 'TopoMotherList_ids'] = stable_charged_df.loc[
        from_primary_selbool, 'ExpandedParticleIndex'].map(decay_dict["topo_ancestors_ids_per_particle"])

    return stable_charged_df


def prcoess_relations(i, relations_buffer, output_data_particles, decay_dict):
    stable_data = output_data_particles[output_data_particles['ParticleType'] == 0].copy()
    particle_index_list = np.sort(stable_data['ExpandedParticleIndex'].values)
    stable_data = stable_data.set_index('ExpandedParticleIndex')

    n = len(particle_index_list)
    idx = np.triu_indices(n, k=1)
    ind_1 = np.array(particle_index_list)[idx[0]]
    ind_2 = np.array(particle_index_list)[idx[1]]

    relations_buffer['FirstParticleIndex'] = ind_1
    relations_buffer['SecondParticleIndex'] = ind_2
    relations_buffer['ExpandedEventNumber'] = i

    data = {"p1": stable_data.loc[ind_1], "p2": stable_data.loc[ind_2]}

    features = ["xProd", "yProd", "zProd", "px", "py", "pz"]
    for particle in data.keys():
        for feature in features:
            relations_buffer[f"{particle}_{feature}"] = data[particle][feature].to_numpy(dtype=np.float64)
            relations_buffer[f"{particle}_{feature}_reco"] = data[particle][f"{feature}_reco"].to_numpy(
                dtype=np.float64)
        for feature in ["MotherList", "MotherList_ids"]:
            relations_buffer[f"{particle}_{feature}"] = data[particle][feature].values
            relations_buffer[f"{particle}_Topo{feature}"] = data[particle][f"Topo{feature}"].values
        relations_buffer[f'{particle}_ParticleRecoType'] = data[particle]['ParticleRecoType'].to_numpy(dtype=np.int32)
        relations_buffer[f'{particle}_isCharged'] = data[particle]['isCharged'].to_numpy(dtype=np.int32)

    p1_hb_index = data["p1"]['ExpandedPrimaryHeavyHadronIndex'].to_numpy(dtype=np.int32)
    p2_hb_index = data["p2"]['ExpandedPrimaryHeavyHadronIndex'].to_numpy(dtype=np.int32)

    relations_buffer['FromSameHeavyHadron'] = np.array(
        (p1_hb_index > 0) * (p2_hb_index > 0) * (p1_hb_index == p2_hb_index), dtype=np.int32)
    relations_buffer['BothInGeomAcc'] = data["p1"]['inGeomAcc'].to_numpy(dtype=np.int32) * data["p2"][
        'inGeomAcc'].to_numpy(
        dtype=np.int32)
    relations_buffer['FromSamePV'] = np.array((data["p1"]['zPV'].to_numpy() == data["p2"]['zPV'].to_numpy()),
                                              dtype=np.int32)
    relations_buffer['FromSameAssociatedPV_reco'] = np.array(
        (data["p1"]['associated_zPV_reco'].to_numpy() == data["p2"]['associated_zPV_reco'].to_numpy()), dtype=np.int32)

    for i in np.unique(data["p1"]['ExpandedPrimaryHeavyHadronIndex']):
        if i not in decay_dict["max_decay_chain_lengths"].keys():
            decay_dict["max_decay_chain_lengths"][i] = 0
            decay_dict["topo_max_decay_chain_lengths"][i] = 0

    data["p1"]['p1_MaxDecayChainLength'] = data["p1"]['ExpandedPrimaryHeavyHadronIndex'].map(
        decay_dict["max_decay_chain_lengths"])
    data["p1"]['p1_TopoMaxDecayChainLength'] = data["p1"]['ExpandedPrimaryHeavyHadronIndex'].map(
        decay_dict["topo_max_decay_chain_lengths"])

    relations_buffer['MaxDecayChainLength'] = relations_buffer['FromSameHeavyHadron'] * data["p1"][
        'p1_MaxDecayChainLength'].to_numpy(dtype=np.int32)
    relations_buffer['TopoMaxDecayChainLength'] = relations_buffer['FromSameHeavyHadron'] * data["p1"][
        'p1_TopoMaxDecayChainLength'].to_numpy(dtype=np.int32)

    relations_buffer = calculate_LCA(relations_buffer, '')
    relations_buffer = calculate_LCA(relations_buffer, 'Topo')
    relations_buffer = calculate_doca(relations_buffer, 0)
    relations_buffer = calculate_doca(relations_buffer, 1)
    relations_buffer = calculate_transversedist(relations_buffer, 0)
    relations_buffer = calculate_transversedist(relations_buffer, 1)
    relations_buffer = calculate_angle(relations_buffer, 0)
    relations_buffer = calculate_angle(relations_buffer, 1)

    return relations_buffer


def format_relations(df):
    df = df.astype({
        "BothInGeomAcc": "bool",
        "DOCA": "float32",
        "DOCA_reco": "float32",
        "ExpandedEventNumber": "int32",
        "FirstParticleIndex": "int32",
        "SecondParticleIndex": "int32",
        "FromSameAssociatedPV_reco": "bool",
        "FromSameHeavyHadron": "bool",
        "FromSamePV": "bool",
        "LCA": "int8",
        "LCA_class0": "bool",
        "LCA_class1": "bool",
        "LCA_class2": "bool",
        "LCA_class3": "bool",
        "LCA_class4": "bool",
        "LCA_class5": "bool",
        "LCA_class6": "bool",
        "LCA_class7": "bool",
        "LCA_class8": "bool",
        "LCA_class9": "bool",
        "LCA_class10": "bool",
        "LCA_ids": "int16",
        "MaxDecayChainLength": "int8",
        "TopoLCA": "int8",
        "TopoLCA_class0": "bool",
        "TopoLCA_class1": "bool",
        "TopoLCA_class2": "bool",
        "TopoLCA_class3": "bool",
        "TopoLCA_class4": "bool",
        "TopoLCA_class5": "bool",
        "TopoLCA_class6": "bool",
        "TopoLCA_class7": "bool",
        "TopoLCA_class8": "bool",
        "TopoLCA_class9": "bool",
        "TopoLCA_class10": "bool",
        "TopoLCA_ids": "int16",
        "TopoMaxDecayChainLength": "int8",
        "delta_z0": "float32",
        "delta_z0_reco": "float32",
        "p1_ParticleRecoType": "int8",
        "p2_ParticleRecoType": "int8",
        "theta": "float32",
        "theta_reco": "float32",
        "trdist": "float32",
        "trdist_reco": "float32"
    })
    return df
